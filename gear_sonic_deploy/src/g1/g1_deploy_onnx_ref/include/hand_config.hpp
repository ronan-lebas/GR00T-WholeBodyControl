/**
 * @file hand_config.hpp
 * @brief Compile-time hand type selection.
 *
 * ## Switching hand types
 *
 * Set USE_BRAINCO_HANDS to 1 for BrainCo Revo2 hands, 0 for Unitree Dex3 hands,
 * then recompile.  No other source files need to be edited.
 *
 * ## BrainCo deployment note
 *
 * When USE_BRAINCO_HANDS = 1, the brainco_hand_service bridge process must be
 * running before starting the deploy binary.  It handles the RS485/Modbus
 * connection to the hardware and exposes the hand via DDS topics:
 *   rt/brainco/{left|right}/{cmd|state}
 */

#pragma once

// *** CHANGE THIS LINE TO 1 FOR BRAINCO HANDS, THEN RECOMPILE ***
#define USE_BRAINCO_HANDS 1

#if USE_BRAINCO_HANDS
static constexpr int NUM_HAND_MOTORS = 6;   ///< BrainCo Revo2: 6 DOF per hand
#else
static constexpr int NUM_HAND_MOTORS = 7;   ///< Unitree Dex3: 7 DOF per hand
#endif
