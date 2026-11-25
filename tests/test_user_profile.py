"""Unit tests for P4 User Profile, Cold-Start & Behavior Adaptation (`src/recommendation/user_profile.py`)."""

from __future__ import annotations

import unittest

from src.contract_validator import validate_scoring_weights
from src.embedder import create_embedder
from src.recommendation.user_profile import (
    BehaviorSignal,
    OnboardingInput,
    UserProfileManager,
    UserProfileState,
)
from src.schemas import ScoringWeights
from src.vector_ops import cosine_similarity, l2_norm


class TestUserProfileManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.embedder, _ = create_embedder()
        cls.manager = UserProfileManager(embedder=cls.embedder)

        cls.vec_ai = cls.embedder.encode_query(
            "Cuộc thi An toàn thông tin HCMUS CTF lập trình AI Machine Learning Khoa CNTT"
        )
        cls.vec_vol = cls.embedder.encode_query(
            "Chiến dịch tình nguyện Mùa Hè Xanh hiến máu quyên góp vì cộng đồng"
        )

    def test_cold_start_e5_vector_alignment_and_empty_case(self) -> None:
        state = self.manager.create_cold_start_profile(
            OnboardingInput(
                user_id="sv_001",
                interest_chips=["ai_tech"],
                cohort="K23",
                faculty="Khoa CNTT",
                home_campus="cs1_nvc",
            )
        )
        self.assertIsNotNone(state.u_declared)
        self.assertIsNotNone(state.profile.user_vec)
        assert state.profile.user_vec is not None
        self.assertEqual(len(state.profile.user_vec), 384)
        self.assertAlmostEqual(l2_norm(state.profile.user_vec), 1.0, places=5)

        sim_ai = cosine_similarity(state.profile.user_vec, self.vec_ai)
        sim_vol = cosine_similarity(state.profile.user_vec, self.vec_vol)
        self.assertGreater(sim_ai, sim_vol)

        # Empty cold start must succeed cleanly without forcing a vector
        empty_state = self.manager.create_cold_start_profile(
            OnboardingInput(user_id="sv_empty")
        )
        self.assertIsNone(empty_state.u_declared)
        self.assertIsNone(empty_state.profile.user_vec)
        self.assertTrue(validate_scoring_weights(empty_state.profile.weights).is_valid)

    def test_preset_and_custom_weights_normalization(self) -> None:
        preset_state = self.manager.create_cold_start_profile(
            OnboardingInput(user_id="sv_preset", weight_preset="need_points")
        )
        self.assertEqual(preset_state.weight_preset, "need_points")
        self.assertAlmostEqual(preset_state.profile.weights.w1_gain, 0.55, places=6)
        self.assertTrue(validate_scoring_weights(preset_state.profile.weights).is_valid)

        custom_state = self.manager.create_cold_start_profile(
            OnboardingInput(
                user_id="sv_custom",
                custom_weights=ScoringWeights(
                    w1_gain=2.0,
                    w2_interest=2.0,
                    w3_fit=1.0,
                    w4_urgency=1.0,
                ),
            )
        )
        self.assertEqual(custom_state.weight_preset, "custom")
        self.assertAlmostEqual(custom_state.profile.weights.total(), 1.0, places=6)
        self.assertAlmostEqual(custom_state.profile.weights.w1_gain, 2.0 / 6.0, places=6)
        self.assertTrue(validate_scoring_weights(custom_state.profile.weights).is_valid)

    def test_behavior_vector_adaptation_and_reset(self) -> None:
        state = self.manager.create_cold_start_profile(
            OnboardingInput(user_id="sv_vec", interest_chips=["ai_tech"])
        )
        assert state.profile.user_vec is not None
        sim_vol_before = cosine_similarity(state.profile.user_vec, self.vec_vol)

        self.manager.record_behavior(
            state,
            BehaviorSignal(event_id="evt_vol_1", action="joined", event_vector=self.vec_vol),
        )
        self.assertIn("evt_vol_1", state.profile.participated_event_ids)
        self.assertIsNotNone(state.u_learned)
        assert state.profile.user_vec is not None
        self.assertAlmostEqual(l2_norm(state.profile.user_vec), 1.0, places=5)
        sim_vol_after_join = cosine_similarity(state.profile.user_vec, self.vec_vol)
        self.assertGreater(sim_vol_after_join, sim_vol_before)

        # Dismiss pushes vector away from dismissed event
        self.manager.record_behavior(
            state,
            BehaviorSignal(event_id="evt_vol_2", action="dismiss", event_vector=self.vec_vol),
        )
        self.assertIn("evt_vol_2", state.profile.hidden_event_ids)
        sim_vol_after_dismiss = cosine_similarity(state.profile.user_vec, self.vec_vol)
        self.assertLess(sim_vol_after_dismiss, sim_vol_after_join)

        # Reset restores cold-start vector and declared weights
        self.manager.reset_learned_behavior(state)
        self.assertIsNone(state.u_learned)
        self.assertEqual(state.behavior_count, 0)
        self.assertEqual(state.profile.user_vec, state.u_declared)

    def test_behavior_driven_weight_adaptation_and_bounds(self) -> None:
        # 1. Repeated joined on high-gain events -> increases w1_gain
        s_gain = self.manager.create_cold_start_profile(
            OnboardingInput(user_id="sv_w_gain", weight_preset="balanced")
        )
        init_w1 = s_gain.profile.weights.w1_gain
        for idx in range(5):
            self.manager.record_behavior(
                s_gain,
                BehaviorSignal(
                    event_id=f"eg_{idx}",
                    action="joined",
                    score_components={"gain": 0.95, "interest": 0.20, "fit": 0.45, "urgency": 0.20},
                ),
            )
            w = s_gain.profile.weights
            self.assertTrue(validate_scoring_weights(w).is_valid)
            for val in (w.w1_gain, w.w2_interest, w.w3_fit, w.w4_urgency):
                self.assertGreaterEqual(val, 0.05 - 1e-9)
                self.assertLessEqual(val, 0.70 + 1e-9)
        self.assertGreater(s_gain.profile.weights.w1_gain, init_w1)

        # 2. Repeated joined on high-interest events -> increases w2_interest
        s_int = self.manager.create_cold_start_profile(
            OnboardingInput(user_id="sv_w_int", weight_preset="balanced")
        )
        init_w2 = s_int.profile.weights.w2_interest
        for idx in range(5):
            self.manager.record_behavior(
                s_int,
                BehaviorSignal(
                    event_id=f"ei_{idx}",
                    action="joined",
                    score_components={"gain": 0.25, "interest": 0.95, "fit": 0.45, "urgency": 0.15},
                ),
            )
            self.assertTrue(validate_scoring_weights(s_int.profile.weights).is_valid)
        self.assertGreater(s_int.profile.weights.w2_interest, init_w2)

        # 3. Dismiss due to schedule_conflict -> increases w3_fit
        s_fit = self.manager.create_cold_start_profile(
            OnboardingInput(user_id="sv_w_fit", weight_preset="balanced")
        )
        init_w3 = s_fit.profile.weights.w3_fit
        self.manager.record_behavior(
            s_fit,
            BehaviorSignal(
                event_id="ef_1",
                action="dismiss",
                dismiss_reason="schedule_conflict",
            ),
        )
        self.assertTrue(validate_scoring_weights(s_fit.profile.weights).is_valid)
        self.assertGreater(s_fit.profile.weights.w3_fit, init_w3)

    def test_state_serialization_roundtrip(self) -> None:
        state = self.manager.create_cold_start_profile(
            OnboardingInput(
                user_id="sv_ser",
                interest_chips=["ai_tech", "soft_skills"],
                weight_preset="interest_first",
                cohort="K23",
                home_campus="cs2_lt",
            )
        )
        self.manager.record_behavior(
            state,
            BehaviorSignal(
                event_id="ev_ser_1",
                action="save",
                event_vector=self.vec_ai,
                score_components={"gain": 0.6, "interest": 0.9, "fit": 0.7, "urgency": 0.4},
            ),
        )
        restored = UserProfileState.from_dict(state.to_dict())
        self.assertEqual(restored.profile.user_id, "sv_ser")
        self.assertEqual(restored.home_campus, "cs2_lt")
        self.assertEqual(restored.weight_preset, "interest_first")
        self.assertEqual(restored.behavior_count, 1)
        self.assertEqual(restored.profile.saved_event_ids, ["ev_ser_1"])
        self.assertAlmostEqual(
            restored.profile.weights.w2_interest,
            state.profile.weights.w2_interest,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
