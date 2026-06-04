/**
 * @file brainco_hands.hpp
 * @brief Driver for BrainCo Revo2 robotic hands (left + right).
 *
 * BraincoHands communicates with the hands via DDS topics published/subscribed
 * by the brainco_hand_service bridge process (must be running separately).
 * It provides the same high-level interface as Dex3Hands so the two classes
 * are interchangeable via the HandDriver type alias in hand_config.hpp.
 *
 * ## Architecture
 *
 * This class does NOT talk to the hardware directly.  The brainco_hand_service
 * process owns the RS485/Modbus connection and translates between DDS and the
 * BrainCo C SDK (libbc_stark_sdk).  This class is the DDS client side.
 *
 * ## DDS Topics
 *
 *   Direction | Left hand                  | Right hand
 *   ----------|----------------------------|---------------------------
 *   Command   | rt/brainco/left/cmd        | rt/brainco/right/cmd
 *   State     | rt/brainco/left/state      | rt/brainco/right/state
 *
 * ## Value Convention
 *
 * All positions and speeds are **normalized to [0.0, 1.0]**:
 *   - q = 0.0  → fully open
 *   - q = 1.0  → fully closed
 *   - dq = 0.0 → stopped, dq = 1.0 → full closing speed (default)
 *
 * The 6 fingers in order: [Thumb, Thumb_aux, Index, Middle, Ring, Pinky]
 *
 * ## Smoothing
 *
 * writeOnce() clamps the per-joint position delta to MAX_DELTA_Q per call
 * (in normalized units) to avoid sudden jumps when the command changes.
 *
 * ## Interface Compatibility with Dex3Hands
 *
 * setAllJointsCommand() accepts a 7-element array (same as the Dex3 call site
 * buffer) but only reads the first BRAINCO_MOTOR_MAX = 6 elements.
 * SetMaxCloseRatio() / GetMaxCloseRatio() are no-ops (BrainCo uses normalized
 * values so no separate close-ratio limit is needed).
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "utils.hpp"

static constexpr int BRAINCO_MOTOR_MAX = 6;  ///< Number of actuated motors per BrainCo hand.

/**
 * @class BraincoHands
 * @brief DDS client driver for two BrainCo Revo2 hands (left + right).
 *
 * Does not run its own thread – the owning class calls writeOnce() at the
 * desired cadence from the command-writer thread (typically 500 Hz).
 */
class BraincoHands {
    // Type aliases declared before `public:` so they are in scope for all
    // public method return types (C++ return-type lookup is at point of declaration).
    using MotorCmds_   = unitree_go::msg::dds_::MotorCmds_;
    using MotorStates_ = unitree_go::msg::dds_::MotorStates_;

public:
    BraincoHands() = default;

    /**
     * @brief Initialize DDS channels for both hands.
     * @param networkInterface  Network interface name (e.g. "eth0").
     *                          Pass an empty string to skip ChannelFactory init
     *                          (already done by the main application).
     */
    void initialize(const std::string& networkInterface) {
        if (!networkInterface.empty()) {
            unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface.c_str());
        }

        // Initialize command buffers with safe defaults (fully open, full speed)
        MotorCmds_ left_cmd;  sizeCommand(left_cmd);
        MotorCmds_ right_cmd; sizeCommand(right_cmd);
        left_.cmd_buffer.SetData(left_cmd);
        right_.cmd_buffer.SetData(right_cmd);

        // Left hand channels
        left_.publisher.reset(
            new unitree::robot::ChannelPublisher<MotorCmds_>("rt/brainco/left/cmd"));
        left_.subscriber.reset(
            new unitree::robot::ChannelSubscriber<MotorStates_>("rt/brainco/left/state"));
        left_.publisher->InitChannel();
        left_.subscriber->InitChannel(
            [this](const void* msg) { this->onState(true, msg); }, 1);

        // Right hand channels
        right_.publisher.reset(
            new unitree::robot::ChannelPublisher<MotorCmds_>("rt/brainco/right/cmd"));
        right_.subscriber.reset(
            new unitree::robot::ChannelSubscriber<MotorStates_>("rt/brainco/right/state"));
        right_.publisher->InitChannel();
        right_.subscriber->InitChannel(
            [this](const void* msg) { this->onState(false, msg); }, 1);
    }

    /**
     * @brief Publish one command tick with delta-q smoothing.
     *
     * Call this at your desired cadence (typically 500 Hz) from the
     * command-writer thread.  Clamps per-joint position deltas to MAX_DELTA_Q
     * to prevent sudden jumps.
     */
    void writeOnce() {
        // Max position change per tick in normalized units.
        // 0.05 / 500 Hz = 0.1 normalized/s → ~10 s to fully close (conservative).
        constexpr double MAX_DELTA_Q = 0.05;

        for (bool is_left : {true, false}) {
            HandCtx& ctx = is_left ? left_ : right_;
            const auto cmdPtr   = ctx.cmd_buffer.GetDataWithTime().data;
            const auto statePtr = ctx.state_buffer.GetDataWithTime().data;

            if (!ctx.publisher || !cmdPtr) { continue; }

            MotorCmds_ smoothedCmd = *cmdPtr;

            // Clamp desired positions to valid [0, 1] range
            for (int i = 0; i < BRAINCO_MOTOR_MAX; ++i) {
                const double desired = std::clamp(static_cast<double>(cmdPtr->cmds()[i].q()), 0.0, 1.0);
                smoothedCmd.cmds()[i].q(static_cast<float>(desired));
            }

            // Apply delta-q smoothing if state feedback is available
            if (statePtr && static_cast<int>(statePtr->states().size()) == BRAINCO_MOTOR_MAX) {
                for (int i = 0; i < BRAINCO_MOTOR_MAX; ++i) {
                    const double current = static_cast<double>(statePtr->states()[i].q());
                    const double desired = static_cast<double>(smoothedCmd.cmds()[i].q());
                    const double delta   = std::clamp(desired - current, -MAX_DELTA_Q, MAX_DELTA_Q);
                    smoothedCmd.cmds()[i].q(static_cast<float>(current + delta));
                }
            }

            ctx.publisher->Write(smoothedCmd);
        }
    }

    /**
     * @brief Set position targets for all joints of one hand.
     *
     * Accepts a 7-element array for call-site compatibility with Dex3Hands
     * (the shared left_hand_joint_buffer_ has size 7).  Only the first
     * BRAINCO_MOTOR_MAX = 6 elements are used.
     *
     * @param is_left   true → left hand, false → right hand.
     * @param q         Normalized position targets [0=open, 1=closed], first 6 used.
     * @param dq        Optional normalized speed targets [0=stopped, 1=full speed].
     *                  Defaults to 1.0 (full speed) if not provided.
     */
    void setAllJointsCommand(bool is_left,
                             const std::array<double, 7>& q,
                             std::optional<std::array<double, 7>> dq = std::nullopt) {
        HandCtx& ctx = is_left ? left_ : right_;
        const auto currentPtr = ctx.cmd_buffer.GetDataWithTime().data;
        MotorCmds_ cmd = currentPtr ? *currentPtr : MotorCmds_();
        if (!currentPtr) { sizeCommand(cmd); }

        for (int i = 0; i < BRAINCO_MOTOR_MAX; ++i) {
            cmd.cmds()[i].q(static_cast<float>(q[i]));
            if (dq) { cmd.cmds()[i].dq(static_cast<float>((*dq)[i])); }
        }
        ctx.cmd_buffer.SetData(std::move(cmd));
    }

    /**
     * @brief Return the latest state snapshot for one hand.
     * @param is_left  true → left hand, false → right hand.
     * @return Shared pointer to the latest MotorStates_ message, or nullptr if
     *         no state has been received yet.
     */
    std::shared_ptr<const MotorStates_> getState(bool is_left) const {
        const HandCtx& ctx = is_left ? left_ : right_;
        return ctx.state_buffer.GetDataWithTime().data;
    }

    /** @brief Move all fingers to the fully-open position (q = 0). */
    void open(bool is_left) { setUniform(is_left, 0.0f); }

    /** @brief Move all fingers to the fully-closed position (q = 1). */
    void close(bool is_left) { setUniform(is_left, 1.0f); }

    /** @brief Hold current position (no-op: keep current command buffer). */
    void hold(bool is_left) { (void)is_left; }

    /** @brief Stop / relax all fingers (equivalent to open for BrainCo). */
    void stop(bool is_left) { open(is_left); }

    // ------------------------------------------------------------------
    // Dex3-compatible no-ops (close-ratio limiting is not applicable to
    // BrainCo normalized [0,1] control).
    // ------------------------------------------------------------------
    void   SetMaxCloseRatio(double) {}
    double GetMaxCloseRatio() const { return 1.0; }

private:
    struct HandCtx {
        unitree::robot::ChannelPublisherPtr<MotorCmds_>   publisher;
        unitree::robot::ChannelSubscriberPtr<MotorStates_> subscriber;
        DataBuffer<MotorStates_> state_buffer;
        DataBuffer<MotorCmds_>  cmd_buffer;
    };

    /** @brief Resize and zero-initialise a command message for 6 fingers. */
    static void sizeCommand(MotorCmds_& cmd) {
        cmd.cmds().resize(BRAINCO_MOTOR_MAX);
        for (int i = 0; i < BRAINCO_MOTOR_MAX; ++i) {
            cmd.cmds()[i].q(0.0f);    // fully open
            cmd.cmds()[i].dq(1.0f);   // full speed (closing direction)
            cmd.cmds()[i].kp(0.0f);
            cmd.cmds()[i].kd(0.0f);
            cmd.cmds()[i].tau(0.0f);
        }
    }

    /** @brief Set all joints to a uniform normalized position. */
    void setUniform(bool is_left, float q_val) {
        HandCtx& ctx = is_left ? left_ : right_;
        const auto currentPtr = ctx.cmd_buffer.GetDataWithTime().data;
        MotorCmds_ cmd = currentPtr ? *currentPtr : MotorCmds_();
        if (!currentPtr) { sizeCommand(cmd); }
        for (int i = 0; i < BRAINCO_MOTOR_MAX; ++i) { cmd.cmds()[i].q(q_val); }
        ctx.cmd_buffer.SetData(std::move(cmd));
    }

    /** @brief DDS subscriber callback – stores incoming state in the buffer. */
    void onState(bool is_left, const void* message) {
        HandCtx& ctx = is_left ? left_ : right_;
        const auto* incoming = static_cast<const MotorStates_*>(message);
        ctx.state_buffer.SetData(*incoming);
    }

    HandCtx left_;
    HandCtx right_;
};
