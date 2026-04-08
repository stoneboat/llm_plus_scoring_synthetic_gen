"""
Tests for the Phase 3.5 privacy-layer architecture.

Contracts protected here:
Bounds
  - ZCDPBound and ApproxDPBound are immutable and validated
  - ZCDPBound.compose and .scale are numerically correct
  - ApproxDPBound rejects invalid inputs

Events
  - PrivacyEvent.is_private distinguishes rho>0 from rho==0
  - CompositeEvent.total_rho sums correctly
  - CompositeEvent.n_private / n_public counts correctly
  - public_token_event() produces rho=0
  - private_token_event() produces rho matching rho_per_token(...)

Analyses — mechanism-specific formulas
  - rho_per_token parity with old compute_rho_per_token
  - total_rho parity with old compute_total_rho
  - token_bound.rho == rho_per_token
  - SVT disabled → rho_svt = 0

Conversions
  - zcdp_to_approx_dp matches old implementation numerically
  - bound_to_approx_dp returns an ApproxDPBound with correct fields
  - zcdp_to_dp_tight falls back to zcdp_to_approx_dp

Accountant — ZCDPAccountant
  - compose accumulates rho correctly
  - compose_many is equivalent to individual composes
  - zero-cost public-token events do not change rho
  - epsilon_at_delta matches manual conversion
  - n_private / n_public counts correctly
  - reset clears state

Planning
  - compute_epsilon parity with old implementation
  - compute_max_private_tokens parity with old implementation
  - planning monotonicity (larger epsilon → more tokens)
  - round-trip: epsilon(max_tokens(target)) ≤ target * 1.05

Reporting
  - privacy_report dict has expected keys and matches old output
  - privacy_metadata dict has expected keys

Compatibility
  - src.privacy_accounting re-exports all legacy symbols
  - src.config.compute_max_private_tokens wrapper parity
  - src.privacy flat import surface works
"""

import math
import pytest

# ---------------------------------------------------------------------------
# Helpers — canonical parameters
# ---------------------------------------------------------------------------

C = 10.0      # clip_bound
S = 255       # batch_size
TAU = 2.0     # temperature
SIGMA = 0.2   # svt_noise
DELTA = 1e-5
EPS = 1.0


# ===========================================================================
# BOUNDS
# ===========================================================================

class TestZCDPBound:
    def test_creation(self):
        from src.privacy.bounds import ZCDPBound
        b = ZCDPBound(rho=0.01)
        assert b.rho == 0.01

    def test_zero_rho_is_valid(self):
        from src.privacy.bounds import ZCDPBound
        b = ZCDPBound(rho=0.0)
        assert b.rho == 0.0

    def test_negative_rho_raises(self):
        from src.privacy.bounds import ZCDPBound
        with pytest.raises(ValueError):
            ZCDPBound(rho=-0.01)

    def test_frozen(self):
        from src.privacy.bounds import ZCDPBound
        b = ZCDPBound(rho=0.05)
        with pytest.raises(Exception):
            b.rho = 0.1  # type: ignore

    def test_compose_adds_rho(self):
        from src.privacy.bounds import ZCDPBound
        b1 = ZCDPBound(rho=0.1)
        b2 = ZCDPBound(rho=0.3)
        composed = b1.compose(b2)
        assert abs(composed.rho - 0.4) < 1e-12

    def test_scale_multiplies_rho(self):
        from src.privacy.bounds import ZCDPBound
        b = ZCDPBound(rho=0.05)
        scaled = b.scale(4)
        assert abs(scaled.rho - 0.2) < 1e-12

    def test_scale_zero_gives_zero(self):
        from src.privacy.bounds import ZCDPBound
        b = ZCDPBound(rho=0.05)
        assert b.scale(0).rho == 0.0


class TestApproxDPBound:
    def test_creation(self):
        from src.privacy.bounds import ApproxDPBound
        b = ApproxDPBound(epsilon=1.0, delta=1e-5)
        assert b.epsilon == 1.0
        assert b.delta == 1e-5

    def test_negative_epsilon_raises(self):
        from src.privacy.bounds import ApproxDPBound
        with pytest.raises(ValueError):
            ApproxDPBound(epsilon=-0.1, delta=1e-5)

    def test_delta_out_of_range_raises(self):
        from src.privacy.bounds import ApproxDPBound
        with pytest.raises(ValueError):
            ApproxDPBound(epsilon=1.0, delta=1.5)
        with pytest.raises(ValueError):
            ApproxDPBound(epsilon=1.0, delta=-0.1)

    def test_frozen(self):
        from src.privacy.bounds import ApproxDPBound
        b = ApproxDPBound(epsilon=1.0, delta=1e-5)
        with pytest.raises(Exception):
            b.epsilon = 2.0  # type: ignore


# ===========================================================================
# EVENTS
# ===========================================================================

class TestPrivacyEvent:
    def test_zero_cost_is_not_private(self):
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound
        e = PrivacyEvent(bound=ZCDPBound(rho=0.0))
        assert not e.is_private

    def test_positive_cost_is_private(self):
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound
        e = PrivacyEvent(bound=ZCDPBound(rho=0.01))
        assert e.is_private

    def test_carries_bound_object(self):
        """Event stores a PrivacyBound, not a raw scalar."""
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound, PrivacyBound
        e = PrivacyEvent(bound=ZCDPBound(rho=0.05))
        assert isinstance(e.bound, PrivacyBound)
        assert isinstance(e.bound, ZCDPBound)

    def test_as_bound_returns_same_object(self):
        """as_bound() returns the bound object, not a copy."""
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound
        b = ZCDPBound(rho=0.05)
        e = PrivacyEvent(bound=b)
        assert e.as_bound() is b

    def test_rho_compat_property_for_zcdp(self):
        """The .rho property still works for ZCDPBound events."""
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound
        e = PrivacyEvent(bound=ZCDPBound(rho=0.05))
        assert e.rho == 0.05

    def test_rho_compat_property_raises_for_non_zcdp(self):
        """The .rho property raises TypeError for non-ZCDPBound events."""
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ApproxDPBound
        e = PrivacyEvent(bound=ApproxDPBound(epsilon=1.0, delta=1e-5))
        with pytest.raises(TypeError):
            _ = e.rho

    def test_is_private_driven_by_bound_is_trivial(self):
        """is_private must use bound.is_trivial, not a raw rho check."""
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ApproxDPBound
        # An approx-DP event with epsilon=0 is trivial
        e_trivial = PrivacyEvent(bound=ApproxDPBound(epsilon=0.0, delta=1e-5))
        assert not e_trivial.is_private
        # An approx-DP event with epsilon>0 is private
        e_private = PrivacyEvent(bound=ApproxDPBound(epsilon=1.0, delta=1e-5))
        assert e_private.is_private

    def test_label_optional(self):
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound
        e = PrivacyEvent(bound=ZCDPBound(rho=0.0))
        assert e.label == ""
        e2 = PrivacyEvent(bound=ZCDPBound(rho=0.0), label="public_token")
        assert e2.label == "public_token"

    def test_frozen(self):
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound
        e = PrivacyEvent(bound=ZCDPBound(rho=0.1))
        with pytest.raises(Exception):
            e.bound = ZCDPBound(rho=0.2)  # type: ignore


class TestCompositeEvent:
    def _zcdp_event(self, rho, label=""):
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound
        return PrivacyEvent(bound=ZCDPBound(rho=rho), label=label)

    def test_total_rho_sum(self):
        from src.privacy.events import CompositeEvent
        events = (
            self._zcdp_event(0.1, "private"),
            self._zcdp_event(0.0, "public"),
            self._zcdp_event(0.2, "private"),
        )
        ce = CompositeEvent(events=events)
        assert abs(ce.total_rho - 0.3) < 1e-12

    def test_n_private_count(self):
        from src.privacy.events import CompositeEvent
        events = (
            self._zcdp_event(0.1),
            self._zcdp_event(0.0),
            self._zcdp_event(0.0),
            self._zcdp_event(0.05),
        )
        ce = CompositeEvent(events=events)
        assert ce.n_private == 2
        assert ce.n_public == 2

    def test_empty_composite_has_zero_rho(self):
        from src.privacy.events import CompositeEvent
        ce = CompositeEvent(events=())
        assert ce.total_rho == 0.0

    def test_as_bound_via_bound_objects(self):
        """as_bound() composes via bound objects, not by summing event.rho directly."""
        from src.privacy.events import CompositeEvent
        from src.privacy.bounds import ZCDPBound
        ce = CompositeEvent(events=(self._zcdp_event(0.05), self._zcdp_event(0.05)))
        b = ce.as_bound()
        assert isinstance(b, ZCDPBound)
        assert abs(b.rho - 0.1) < 1e-12

    def test_compose_concatenates(self):
        from src.privacy.events import CompositeEvent
        ce1 = CompositeEvent(events=(self._zcdp_event(0.1),))
        ce2 = CompositeEvent(events=(self._zcdp_event(0.0),))
        combined = ce1.compose(ce2)
        assert len(combined.events) == 2
        assert abs(combined.total_rho - 0.1) < 1e-12


# ===========================================================================
# ANALYSES — private_prediction
# ===========================================================================

class TestPrivatePredictionAnalysis:
    def test_rho_per_token_no_svt_parity(self):
        """Must match old compute_rho_per_token."""
        from src.privacy.analyses.private_prediction import rho_per_token
        from src.privacy_accounting import compute_rho_per_token
        assert abs(
            rho_per_token(C, S, TAU, None)
            - compute_rho_per_token(C, S, TAU, None)
        ) < 1e-15

    def test_rho_per_token_with_svt_parity(self):
        from src.privacy.analyses.private_prediction import rho_per_token
        from src.privacy_accounting import compute_rho_per_token
        assert abs(
            rho_per_token(C, S, TAU, SIGMA)
            - compute_rho_per_token(C, S, TAU, SIGMA)
        ) < 1e-15

    def test_total_rho_parity(self):
        from src.privacy.analyses.private_prediction import total_rho
        from src.privacy_accounting import compute_total_rho
        r = 50
        assert abs(
            total_rho(r, C, S, TAU, SIGMA)
            - compute_total_rho(r, C, S, TAU, SIGMA)
        ) < 1e-15

    def test_svt_disabled_rho_svt_term_zero(self):
        """With svt_noise=None, rho == rho_exp only."""
        from src.privacy.analyses.private_prediction import rho_per_token
        rho_no_svt = rho_per_token(C, S, TAU, None)
        rho_exp = 0.5 * (C / (S * TAU)) ** 2
        assert abs(rho_no_svt - rho_exp) < 1e-15

    def test_token_bound_rho_matches_rho_per_token(self):
        from src.privacy.analyses.private_prediction import rho_per_token, token_bound
        expected = rho_per_token(C, S, TAU, SIGMA)
        b = token_bound(C, S, TAU, SIGMA)
        assert abs(b.rho - expected) < 1e-15

    def test_public_token_event_zero_cost(self):
        from src.privacy.analyses.private_prediction import public_token_event
        from src.privacy.bounds import ZCDPBound
        e = public_token_event()
        assert isinstance(e.bound, ZCDPBound)
        assert e.bound.rho == 0.0
        assert not e.is_private

    def test_private_token_event_carries_zcdp_bound(self):
        from src.privacy.analyses.private_prediction import (
            private_token_event, rho_per_token
        )
        from src.privacy.bounds import ZCDPBound
        e = private_token_event(C, S, TAU, SIGMA)
        assert isinstance(e.bound, ZCDPBound)
        assert e.is_private
        assert abs(e.bound.rho - rho_per_token(C, S, TAU, SIGMA)) < 1e-15


# ===========================================================================
# CONVERSIONS
# ===========================================================================

class TestConversions:
    def test_zcdp_to_approx_dp_parity(self):
        from src.privacy.conversions import zcdp_to_approx_dp
        from src.privacy_accounting import zcdp_to_approx_dp as old
        rho = 0.05
        assert abs(zcdp_to_approx_dp(rho, DELTA) - old(rho, DELTA)) < 1e-15

    def test_zcdp_to_approx_dp_zero_rho(self):
        from src.privacy.conversions import zcdp_to_approx_dp
        assert zcdp_to_approx_dp(0.0, DELTA) == 0.0

    def test_zcdp_to_approx_dp_monotone_in_rho(self):
        from src.privacy.conversions import zcdp_to_approx_dp
        eps1 = zcdp_to_approx_dp(0.01, DELTA)
        eps2 = zcdp_to_approx_dp(0.1, DELTA)
        assert eps2 > eps1

    def test_zcdp_to_dp_tight_fallback(self):
        """Should fall back to approx formula (library path is stubbed)."""
        from src.privacy.conversions import zcdp_to_dp_tight, zcdp_to_approx_dp
        rho = 0.05
        assert abs(zcdp_to_dp_tight(rho, DELTA) - zcdp_to_approx_dp(rho, DELTA)) < 1e-15

    def test_bound_to_approx_dp_types(self):
        from src.privacy.conversions import bound_to_approx_dp
        from src.privacy.bounds import ZCDPBound, ApproxDPBound
        b = ZCDPBound(rho=0.05)
        approx = bound_to_approx_dp(b, DELTA)
        assert isinstance(approx, ApproxDPBound)
        assert approx.delta == DELTA

    def test_bound_to_approx_dp_epsilon_matches_formula(self):
        from src.privacy.conversions import bound_to_approx_dp, zcdp_to_approx_dp
        from src.privacy.bounds import ZCDPBound
        b = ZCDPBound(rho=0.05)
        approx = bound_to_approx_dp(b, DELTA)
        expected_eps = zcdp_to_approx_dp(0.05, DELTA)
        assert abs(approx.epsilon - expected_eps) < 1e-15


# ===========================================================================
# ACCOUNTANT — ZCDPAccountant
# ===========================================================================

class TestZCDPAccountant:
    def test_initial_state_zero(self):
        from src.privacy.accountants.zcdp import ZCDPAccountant
        acct = ZCDPAccountant()
        assert acct.total_rho == 0.0
        assert acct.n_events == 0

    def test_compose_private_token(self):
        from src.privacy.accountants.zcdp import ZCDPAccountant
        from src.privacy.analyses.private_prediction import private_token_event
        acct = ZCDPAccountant()
        e = private_token_event(C, S, TAU, SIGMA)
        acct.compose(e)
        assert abs(acct.total_rho - e.bound.rho) < 1e-15
        assert acct.n_private == 1
        assert acct.n_public == 0

    def test_public_token_does_not_increase_rho(self):
        from src.privacy.accountants.zcdp import ZCDPAccountant
        from src.privacy.analyses.private_prediction import (
            private_token_event, public_token_event
        )
        acct = ZCDPAccountant()
        priv_e = private_token_event(C, S, TAU, SIGMA)
        pub_e = public_token_event()
        acct.compose(priv_e)
        rho_after_private = acct.total_rho
        acct.compose(pub_e)
        assert acct.total_rho == rho_after_private, (
            "Public token must not increase rho"
        )
        assert acct.n_public == 1

    def test_compose_many_equivalent_to_individual(self):
        from src.privacy.accountants.zcdp import ZCDPAccountant
        from src.privacy.analyses.private_prediction import (
            private_token_event, public_token_event
        )
        events = [
            private_token_event(C, S, TAU, SIGMA),
            public_token_event(),
            private_token_event(C, S, TAU, None),
        ]
        acct_many = ZCDPAccountant().compose_many(events)
        acct_indiv = ZCDPAccountant()
        for e in events:
            acct_indiv.compose(e)
        assert abs(acct_many.total_rho - acct_indiv.total_rho) < 1e-15

    def test_epsilon_at_delta_matches_formula(self):
        from src.privacy.accountants.zcdp import ZCDPAccountant
        from src.privacy.analyses.private_prediction import private_token_event
        from src.privacy.conversions import zcdp_to_approx_dp
        acct = ZCDPAccountant()
        e = private_token_event(C, S, TAU, SIGMA)
        for _ in range(50):
            acct.compose(e)
        expected_eps = zcdp_to_approx_dp(acct.total_rho, DELTA)
        assert abs(acct.epsilon_at_delta(DELTA) - expected_eps) < 1e-15

    def test_current_bound_type(self):
        from src.privacy.accountants.zcdp import ZCDPAccountant
        from src.privacy.bounds import ZCDPBound
        acct = ZCDPAccountant()
        assert isinstance(acct.current_bound(), ZCDPBound)

    def test_current_bound_rho_matches_total(self):
        from src.privacy.accountants.zcdp import ZCDPAccountant
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound
        acct = ZCDPAccountant()
        acct.compose(PrivacyEvent(bound=ZCDPBound(rho=0.1)))
        acct.compose(PrivacyEvent(bound=ZCDPBound(rho=0.2)))
        assert abs(acct.current_bound().rho - 0.3) < 1e-12

    def test_reset_clears_state(self):
        from src.privacy.accountants.zcdp import ZCDPAccountant
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound
        acct = ZCDPAccountant()
        acct.compose(PrivacyEvent(bound=ZCDPBound(rho=0.1)))
        acct.compose(PrivacyEvent(bound=ZCDPBound(rho=0.1)))
        acct.reset()
        assert acct.total_rho == 0.0
        assert acct.n_events == 0

    def test_compose_rejects_non_zcdp_event(self):
        """ZCDPAccountant must raise TypeError for non-ZCDPBound events."""
        from src.privacy.accountants.zcdp import ZCDPAccountant
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ApproxDPBound
        acct = ZCDPAccountant()
        non_zcdp_event = PrivacyEvent(bound=ApproxDPBound(epsilon=1.0, delta=1e-5))
        with pytest.raises(TypeError):
            acct.compose(non_zcdp_event)

    def test_compose_uses_event_bound_rho_not_event_rho(self):
        """Accountant must extract rho via event.bound.rho (the explicit path)."""
        from src.privacy.accountants.zcdp import ZCDPAccountant
        from src.privacy.events import PrivacyEvent
        from src.privacy.bounds import ZCDPBound
        rho = 0.07
        e = PrivacyEvent(bound=ZCDPBound(rho=rho))
        acct = ZCDPAccountant()
        acct.compose(e)
        # Verify rho came through correctly via the bound path
        assert abs(acct.total_rho - rho) < 1e-15
        assert abs(acct.current_bound().rho - rho) < 1e-15

    def test_accountant_composition_parity_with_planning(self):
        """Accountant with r private tokens should yield same epsilon as planning helper."""
        from src.privacy.accountants.zcdp import ZCDPAccountant
        from src.privacy.analyses.private_prediction import private_token_event
        from src.privacy.planning import compute_epsilon
        r = 50
        acct = ZCDPAccountant()
        e = private_token_event(C, S, TAU, SIGMA)
        acct.compose_many([e] * r)
        eps_acct = acct.epsilon_at_delta(DELTA)
        eps_plan = compute_epsilon(r, C, S, TAU, DELTA, SIGMA)
        assert abs(eps_acct - eps_plan) < 1e-12, (
            f"Accountant eps={eps_acct} != planning eps={eps_plan}"
        )


# ===========================================================================
# PLANNING
# ===========================================================================

class TestPlanning:
    def test_compute_epsilon_parity(self):
        from src.privacy.planning import compute_epsilon
        from src.privacy_accounting import compute_epsilon as old
        r = 100
        assert abs(
            compute_epsilon(r, C, S, TAU, DELTA, SIGMA)
            - old(r, C, S, TAU, DELTA, SIGMA)
        ) < 1e-15

    def test_compute_max_private_tokens_parity(self):
        from src.privacy.planning import compute_max_private_tokens
        from src.privacy_accounting import compute_max_private_tokens as old
        # authoritative order: clip_bound before batch_size
        assert (
            compute_max_private_tokens(EPS, DELTA, C, S, TAU, SIGMA)
            == old(EPS, DELTA, C, S, TAU, SIGMA)
        )

    def test_larger_epsilon_more_tokens(self):
        from src.privacy.planning import compute_max_private_tokens
        r_small = compute_max_private_tokens(0.5, DELTA, C, S, TAU, None)
        r_large = compute_max_private_tokens(5.0, DELTA, C, S, TAU, None)
        assert r_large >= r_small

    def test_larger_batch_size_more_tokens(self):
        from src.privacy.planning import compute_max_private_tokens
        r_small = compute_max_private_tokens(EPS, DELTA, C, 127, TAU, None)
        r_large = compute_max_private_tokens(EPS, DELTA, C, 1023, TAU, None)
        assert r_large >= r_small

    def test_svt_reduces_tokens(self):
        from src.privacy.planning import compute_max_private_tokens
        r_no_svt = compute_max_private_tokens(EPS, DELTA, C, S, TAU, None)
        r_svt = compute_max_private_tokens(EPS, DELTA, C, S, TAU, SIGMA)
        assert r_svt <= r_no_svt

    def test_round_trip_within_target(self):
        from src.privacy.planning import compute_max_private_tokens, compute_epsilon
        r = compute_max_private_tokens(EPS, DELTA, C, S, TAU, SIGMA)
        actual = compute_epsilon(r, C, S, TAU, DELTA, SIGMA)
        assert actual <= EPS * 1.05, (
            f"Round-trip epsilon {actual:.4f} > target {EPS} * 1.05"
        )

    def test_result_is_positive_integer(self):
        from src.privacy.planning import compute_max_private_tokens
        r = compute_max_private_tokens(EPS, DELTA, C, S, TAU, None)
        assert isinstance(r, int) and r >= 1

    def test_tiny_epsilon_returns_one(self):
        from src.privacy.planning import compute_max_private_tokens
        r = compute_max_private_tokens(1e-10, DELTA, C, S, TAU, None)
        assert r >= 1


# ===========================================================================
# REPORTING
# ===========================================================================

class TestReporting:
    def test_privacy_report_parity(self):
        """privacy_report must match old implementation exactly."""
        from src.privacy.reporting import privacy_report
        from src.privacy_accounting import privacy_report as old
        r = 100
        new_rep = privacy_report(r, C, S, TAU, DELTA, SIGMA)
        old_rep = old(r, C, S, TAU, DELTA, SIGMA)
        for key, val in old_rep.items():
            assert key in new_rep, f"Missing key: {key}"
            if isinstance(val, float):
                assert abs(new_rep[key] - val) < 1e-15, (
                    f"Mismatch at '{key}': {new_rep[key]} vs {val}"
                )
            else:
                assert new_rep[key] == val

    def test_privacy_report_required_keys(self):
        from src.privacy.reporting import privacy_report
        rep = privacy_report(100, C, S, TAU, DELTA, SIGMA)
        required = {
            "rho_per_token", "total_rho", "epsilon", "delta",
            "num_private_tokens", "clip_bound", "batch_size",
            "temperature", "svt_noise",
        }
        assert required <= set(rep.keys())

    def test_privacy_metadata_keys(self):
        from src.privacy.reporting import privacy_metadata
        md = privacy_metadata(
            epsilon=EPS, delta=DELTA, batch_size=S, clip_bound=C,
            temperature=TAU, public_temperature=1.5, svt_threshold=0.5,
            svt_noise=SIGMA, top_k_vocab=0, max_private_tokens=100,
        )
        required = {
            "epsilon", "delta", "batch_size", "clip_bound", "temperature",
            "public_temperature", "svt_threshold", "svt_noise",
            "top_k_vocab", "max_private_tokens",
        }
        assert required <= set(md.keys())

    def test_privacy_metadata_values(self):
        from src.privacy.reporting import privacy_metadata
        md = privacy_metadata(
            epsilon=EPS, delta=DELTA, batch_size=S, clip_bound=C,
            temperature=TAU, public_temperature=1.5, svt_threshold=0.5,
            svt_noise=SIGMA, top_k_vocab=0, max_private_tokens=100,
        )
        assert md["epsilon"] == EPS
        assert md["delta"] == DELTA
        assert md["batch_size"] == S
        assert md["max_private_tokens"] == 100
        assert md["svt_noise"] == SIGMA


# ===========================================================================
# COMPATIBILITY PATHS
# ===========================================================================

class TestCompatibility:
    def test_privacy_accounting_facade_exports_all_symbols(self):
        """All legacy symbols must still be importable from privacy_accounting."""
        from src.privacy_accounting import (
            compute_rho_per_token,
            compute_total_rho,
            zcdp_to_approx_dp,
            zcdp_to_dp_tight,
            compute_epsilon,
            compute_max_private_tokens,
            privacy_report,
        )

    def test_config_wrapper_parity(self):
        """src.config.compute_max_private_tokens must match authoritative output."""
        from src.config import compute_max_private_tokens as config_fn
        from src.privacy.planning import compute_max_private_tokens as auth_fn
        # config order: (eps, delta, batch_size, clip_bound, tau, svt)
        # auth order:   (eps, delta, clip_bound, batch_size, tau, svt)
        r_config = config_fn(EPS, DELTA, S, C, TAU, SIGMA)
        r_auth = auth_fn(EPS, DELTA, C, S, TAU, SIGMA)
        assert r_config == r_auth

    def test_config_wrapper_arg_order_is_different(self):
        """Confirm config and auth wrappers have swapped args (not identical)."""
        from src.config import compute_max_private_tokens as config_fn
        from src.privacy.planning import compute_max_private_tokens as auth_fn
        # Use batch_size=255, clip_bound=1.0 via config order
        r_config = config_fn(EPS, DELTA, S, 1.0, TAU, None)
        # Use clip_bound=1.0, batch_size=255 via auth order (same logical input)
        r_auth = auth_fn(EPS, DELTA, 1.0, S, TAU, None)
        assert r_config == r_auth

    def test_src_privacy_flat_import(self):
        """The src.privacy package exposes a flat import surface."""
        from src.privacy import (
            PrivacyBound,
            ZCDPBound, ApproxDPBound,
            PrivacyEvent, CompositeEvent,
            zcdp_to_approx_dp, zcdp_to_dp_tight, bound_to_approx_dp,
            rho_per_token, total_rho, token_bound,
            private_token_event, public_token_event,
            ZCDPAccountant,
            compute_epsilon, compute_max_private_tokens,
            privacy_report, privacy_metadata,
        )

    def test_src_privacy_analyses_module_alias(self):
        from src.privacy import analyses
        assert hasattr(analyses, "private_prediction")

    def test_src_privacy_accountants_importable(self):
        from src.privacy.accountants import PrivacyAccountant, ZCDPAccountant
        assert issubclass(ZCDPAccountant, PrivacyAccountant)

    def test_privacy_bound_is_common_base(self):
        """Both ZCDPBound and ApproxDPBound are PrivacyBound subclasses."""
        from src.privacy.bounds import PrivacyBound, ZCDPBound, ApproxDPBound
        assert issubclass(ZCDPBound, PrivacyBound)
        assert issubclass(ApproxDPBound, PrivacyBound)

    def test_privacy_accountant_current_bound_returns_privacy_bound(self):
        """current_bound() return type is PrivacyBound (not ZCDPBound specifically)."""
        from src.privacy.accountants import PrivacyAccountant, ZCDPAccountant
        from src.privacy.bounds import PrivacyBound, ZCDPBound
        acct = ZCDPAccountant()
        b = acct.current_bound()
        # Concrete type is ZCDPBound
        assert isinstance(b, ZCDPBound)
        # But it IS a PrivacyBound — the interface is more neutral
        assert isinstance(b, PrivacyBound)
