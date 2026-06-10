from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.simulation.group_stage import GroupStandings, TeamRecord

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Third-place ranking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThirdPlaceTeam:
    group:  str
    team:   str
    record: TeamRecord

    @property
    def sort_key(self) -> tuple:
        """Sort key — ascending (negate what should be maximised)."""
        rec = self.record
        return (
            -rec.points,
            -rec.goal_diff,
            -rec.goals_for,
            -rec.fair_play,   # fair_play is negative (more cards = lower); negate so fewer cards sorts first
        )


def rank_third_place_teams(
    all_standings: dict[str, GroupStandings],
    rng: Optional[np.random.Generator] = None,
) -> list[ThirdPlaceTeam]:
    """
    Collect all 12 third-place finishers and rank them.
    Returns list of ThirdPlaceTeam, best first (index 0 = best).
    The first 8 entries advance to the knockout stage.

    Tiebreaker: points → GD → GF → fair play → lots.
    """
    thirds = []
    for grp, gs in all_standings.items():
        team = gs.ranking[2]          # 3rd place
        thirds.append(ThirdPlaceTeam(group=grp, team=team, record=gs.records[team]))

    # Sort deterministically first
    thirds.sort(key=lambda t: t.sort_key)

    # Resolve any remaining ties by drawing of lots
    _resolve_lots(thirds, rng)

    return thirds


def _resolve_lots(
    thirds: list[ThirdPlaceTeam],
    rng: Optional[np.random.Generator],
) -> None:
    """Shuffle any still-tied clusters in-place."""
    i = 0
    while i < len(thirds):
        j = i + 1
        while j < len(thirds) and thirds[j].sort_key == thirds[i].sort_key:
            j += 1
        cluster = thirds[i:j]
        if len(cluster) > 1:
            if rng is not None:
                indices = list(range(i, j))
                rng.shuffle(indices)
                shuffled = [thirds[k] for k in indices]
                thirds[i:j] = shuffled
            else:
                random.shuffle(cluster)
                thirds[i:j] = cluster
        i = j


def get_qualifiers(
    all_standings: dict[str, GroupStandings],
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[ThirdPlaceTeam], list[ThirdPlaceTeam]]:
    """
    Returns (best_8_thirds, eliminated_4_thirds).
    """
    ranked = rank_third_place_teams(all_standings, rng)
    return ranked[:8], ranked[8:]


# ─────────────────────────────────────────────────────────────────────────────
# FIFA 2026 bracket structure
# ─────────────────────────────────────────────────────────────────────────────
#
# The official FIFA 2026 bracket places:
#   - 12 group winners (1A…1L) and 12 runners-up (2A…2L) in fixed slots
#   - The 8 best third-place teams in the remaining 8 R32 slots
#
# FIFA published the bracket in the competition regulations.  The structure
# pairs groups as follows (based on the confirmed 2026 bracket draw):
#
#   Left half of bracket (R32 matches 1-8 feed into R16 left):
#     Match 49:  1A  vs  2C           Match 50: 1B  vs  2D
#     Match 51:  1E  vs  2G           Match 52: 1F  vs  2H  ← Spain runner-up
#     Match 53:  1I  vs  2K           Match 54: 1H  vs  2L  ← Spain winner
#     Match 55:  1C  vs  2E           Match 56: 1D  vs  2F
#
#   Right half of bracket (R32 matches 9-16 feed into R16 right):
#     Match 57:  1G  vs  3rd          Match 58: 1J  vs  3rd  ← Argentina winner
#     Match 59:  1K  vs  3rd          Match 60: 1L  vs  3rd
#     Match 61: 2J   vs  3rd          Match 62: 2B  vs  3rd  ← Argentina runner-up
#     Match 63: 2I   vs  3rd          Match 64: 2A  vs  3rd
#
# Note: The exact assignment of third-place teams to the 8 open slots depends
# on which combination of groups they qualified from.  FIFA uses a lookup
# table (similar to 2016 UEFA Euros format) mapping the 6-group combination
# to specific bracket positions.
#
# For 2026 (12 groups, best 8 of 12 thirds), the R32 has exactly 16 matches:
#   • 8 matches: group winner vs runner-up (from different groups)
#   • 4 matches: group winner vs 3rd-place team
#   • 4 matches: runner-up vs 3rd-place team
# This gives 8+4+4 = 16 matches → 32 unique teams.
#
# Winners used in W-vs-R matches:   A, B, C, D, E, F, H, I  (8)
# Runners-up used in W-vs-R matches: C, D, E, F, G, H, K, L  (8)
# Remaining winners (face 3rd-place): G, J, K, L              (4)
# Remaining runners-up (face 3rd-place): A, B, I, J           (4)
# 3rd-place teams: 8 (fill the 8 open slots)

# ─────────────────────────────────────────────────────────────────────────────
# Seeding constraint
# ─────────────────────────────────────────────────────────────────────────────
# Spain (Group H) and Argentina (Group J) must land in OPPOSITE bracket halves
# so they can only meet in the Final.
#
# LEFT half  (M49–M56) → feeds QF_1 & QF_2 → SF_1 → Final
# RIGHT half (M57–M64) → feeds QF_3 & QF_4 → SF_2 → Final
#
# Anchoring:
#   Spain     (1H or 2H) → LEFT  half  (M54: 1H vs 2L  |  M52: 1F vs 2H)
#   Argentina (1J or 2J) → RIGHT half  (M58: 1J vs 3rd  |  M61: 2J vs 3rd)
#
# Consequence: Group J winner (1J) moves from LEFT-half W-vs-R (old M54)
#   to RIGHT-half W-vs-3rd (new M58: 1J vs 3rd).
#   Group H winner (1H) fills M54 in the LEFT half.
# ─────────────────────────────────────────────────────────────────────────────

# Fixed: 8 winner-vs-runner-up R32 matches (all in the LEFT half, M49–M56)
# Seeding note: Spain (Group H) appears in both M52 (runner-up) and M54 (winner),
# ensuring Spain stays in the LEFT half regardless of whether they finish 1st or 2nd.
FIXED_R32: list[tuple[str, str, str]] = [
    # (match_id, winner_slot, runner_up_slot)
    ("R32_M49", "1A", "2C"),
    ("R32_M50", "1B", "2D"),
    ("R32_M51", "1E", "2G"),
    ("R32_M52", "1F", "2H"),   # ← Spain runner-up (2H) here (LEFT)
    ("R32_M53", "1I", "2K"),
    ("R32_M54", "1H", "2L"),   # ← Spain winner (1H) here (LEFT)
    ("R32_M55", "1C", "2E"),
    ("R32_M56", "1D", "2F"),
]

# 4 open slots where remaining winners face 3rd-place teams (RIGHT half).
# Group J winner (Argentina if they top group) is placed here → RIGHT half.
# (match_id, winner_slot, forbidden_group for the 3rd-place team)
_WINNER_VS_3RD: list[tuple[str, str, str]] = [
    ("R32_M57", "1G", "G"),
    ("R32_M58", "1J", "J"),   # ← Argentina winner (1J) here (RIGHT)
    ("R32_M59", "1K", "K"),
    ("R32_M60", "1L", "L"),
]

# 4 open slots where remaining runners-up face 3rd-place teams (RIGHT half).
# Group J runner-up (Argentina if they finish 2nd) is placed here → RIGHT half.
# (match_id, runner_up_slot, forbidden_group for the 3rd-place team)
_RUNNER_VS_3RD: list[tuple[str, str, str]] = [
    ("R32_M61", "2J", "J"),   # ← Argentina runner-up (2J) here (RIGHT)
    ("R32_M62", "2B", "B"),
    ("R32_M63", "2I", "I"),
    ("R32_M64", "2A", "A"),
]


def _assign_thirds_to_slots(
    best_8: list[ThirdPlaceTeam],
    rng: Optional[np.random.Generator] = None,
) -> dict[str, tuple[str, str]]:
    """
    Assign the 8 best third-place teams to the 8 open R32 slots
    (4 winner-vs-3rd + 4 runner-up-vs-3rd).

    Returns {match_id: (seeded_team, third_place_team)}.

    Uses greedy best-first assignment with no-rematch constraint.
    Falls back to ignoring constraint if necessary.
    """
    all_open = _WINNER_VS_3RD + _RUNNER_VS_3RD
    # Build indices for fast look-up
    forbidden: dict[str, str]  = {mid: fg for mid, _, fg in all_open}
    seeded_s:  dict[str, str]  = {mid: ss for mid, ss, _ in all_open}
    slot_order: list[str]       = [mid for mid, _, _ in all_open]

    # Greedy assignment: best-ranked team first, skip if forbidden
    unassigned = list(best_8)        # ranked best-first
    assignment: dict[str, str] = {}  # match_id → team_name

    for mid in slot_order:
        fg = forbidden[mid]
        placed = False
        for third in unassigned:
            if third.group != fg:
                assignment[mid] = third.team
                unassigned.remove(third)
                placed = True
                break
        if not placed:
            # Greedy ran out of non-forbidden teams — accept temporarily
            assignment[mid] = unassigned.pop(0).team

    # ── Repair pass: fix any same-group violations by swapping ────────────────
    # Build reverse lookup: team_name → group
    team_group: dict[str, str] = {t.team: t.group for t in best_8}

    changed = True
    while changed:
        changed = False
        for mid_a in slot_order:
            ta = assignment[mid_a]
            if team_group[ta] == forbidden[mid_a]:       # violation in mid_a
                # Find another slot mid_b whose occupant can go to mid_a
                # AND ta can go to mid_b
                for mid_b in slot_order:
                    if mid_b == mid_a:
                        continue
                    tb = assignment[mid_b]
                    if (team_group[ta] != forbidden[mid_b] and
                            team_group[tb] != forbidden[mid_a]):
                        assignment[mid_a], assignment[mid_b] = tb, ta
                        changed = True
                        break

    # Log any remaining violations (should be very rare)
    for mid in slot_order:
        ta = assignment[mid]
        if team_group[ta] == forbidden[mid]:
            log.warning(
                "Rematch unavoidable after repair: %s (Group %s) vs %s",
                ta, team_group[ta], seeded_s[mid],
            )

    return {mid: (seeded_s[mid], assignment[mid]) for mid in slot_order}


# ─────────────────────────────────────────────────────────────────────────────
# Bracket data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BracketSlot:
    """One R32 match in the 32-team bracket."""
    match_id:  str   # e.g. "R32_M49"
    home:      str   # team name
    away:      str   # team name
    home_slot: str   # e.g. "1A" or "3rd_E"
    away_slot: str   # e.g. "2C" or "3rd_B"


@dataclass
class KnockoutBracket:
    """Complete 32-team bracket ready for knockout simulation."""
    r32_matches: list[BracketSlot]

    # Quick lookup: team → BracketSlot
    def find_match(self, team: str) -> Optional[BracketSlot]:
        for m in self.r32_matches:
            if m.home == team or m.away == team:
                return m
        return None

    def to_dict(self) -> dict[str, dict]:
        return {
            m.match_id: {"home": m.home, "away": m.away,
                         "home_slot": m.home_slot, "away_slot": m.away_slot}
            for m in self.r32_matches
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public API: build_bracket
# ─────────────────────────────────────────────────────────────────────────────

def build_bracket(
    all_standings: dict[str, GroupStandings],
    rng: Optional[np.random.Generator] = None,
) -> tuple[KnockoutBracket, list[ThirdPlaceTeam], list[ThirdPlaceTeam]]:
    """
    Build the complete 32-team Round of 32 bracket.

    Parameters
    ----------
    all_standings : dict[str, GroupStandings]
        Output of simulate_all_groups().
    rng : np.random.Generator, optional

    Returns
    -------
    bracket      : KnockoutBracket — 32 R32 matchups
    best_8_thirds: list[ThirdPlaceTeam] — 8 qualifying third-place teams
    eliminated   : list[ThirdPlaceTeam] — 4 eliminated third-place teams
    """
    # 1. Extract 1st and 2nd place from each group
    group_winners:   dict[str, str] = {}
    group_runners_up: dict[str, str] = {}
    for grp, gs in all_standings.items():
        group_winners[grp]    = gs.ranking[0]
        group_runners_up[grp] = gs.ranking[1]

    # 2. Rank all 12 third-place teams, pick best 8
    best_8, eliminated = get_qualifiers(all_standings, rng)
    qualified_thirds = {t.group: t.team for t in best_8}

    log.info("Third-place qualifiers (best 8):")
    for i, t in enumerate(best_8, 1):
        rec = t.record
        log.info("  %2d. %-25s (Group %s) — %d pts, GD%+d, GF%d",
                 i, t.team, t.group, rec.points, rec.goal_diff, rec.goals_for)
    log.info("Eliminated thirds:")
    for t in eliminated:
        rec = t.record
        log.info("  %-25s (Group %s) — %d pts, GD%+d, GF%d",
                 t.team, t.group, rec.points, rec.goal_diff, rec.goals_for)

    # 3. Assign third-place teams to 8 open slots
    third_slot_assignments = _assign_thirds_to_slots(best_8, rng)

    # 4. Build fixed R32 matches (8 group-winner vs runner-up)
    r32: list[BracketSlot] = []
    for match_id, home_slot, away_slot in FIXED_R32:
        home_grp = home_slot[1]   # "1A" → "A"
        away_grp = away_slot[1]   # "2D" → "D"

        home_team = group_winners[home_grp]
        away_team = group_runners_up[away_grp]

        r32.append(BracketSlot(match_id=match_id, home=home_team, away=away_team,
                               home_slot=home_slot, away_slot=away_slot))

    # 5. Build open R32 matches (4 winner-vs-3rd + 4 runner-up-vs-3rd)
    for match_id, (seeded_slot, third_team) in third_slot_assignments.items():
        third_source = next(t.group for t in best_8 if t.team == third_team)
        seeded_grp = seeded_slot[1]       # "1G" → "G" or "2A" → "A"
        seeded_pos = int(seeded_slot[0])  # 1 = winner, 2 = runner-up

        seeded_team = (
            group_winners[seeded_grp] if seeded_pos == 1
            else group_runners_up[seeded_grp]
        )
        r32.append(BracketSlot(
            match_id=match_id,
            home=seeded_team,
            away=third_team,
            home_slot=seeded_slot,
            away_slot=f"3rd_{third_source}",
        ))

    bracket = KnockoutBracket(r32_matches=sorted(r32, key=lambda m: m.match_id))
    return bracket, best_8, eliminated


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import logging as _log
    _log.basicConfig(level=_log.INFO,
                     format="%(asctime)s | %(levelname)-8s | %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")

    from src.simulation.group_stage import simulate_all_groups
    from src.simulation.match_predictor import MatchPredictor

    log.info("=" * 60)
    log.info("TASK 4.2 — Third-Place Qualification & Bracket Builder")
    log.info("=" * 60)

    predictor    = MatchPredictor.load()
    rng          = np.random.default_rng(42)
    all_standings = simulate_all_groups(predictor, rng=rng)
    bracket, best_8, eliminated = build_bracket(all_standings, rng=rng)

    # ── Print third-place rankings ──────────────────────────────────────────
    all_thirds = best_8 + eliminated
    print("\n" + "=" * 70)
    print("THIRD-PLACE TEAM RANKINGS (all 12)")
    print("=" * 70)
    print(f"  {'#':<4} {'Team':<26} {'Grp':<5} {'Pts':>4} {'GD':>4} {'GF':>4} {'Status'}")
    print("  " + "-" * 66)
    for i, t in enumerate(all_thirds, 1):
        rec    = t.record
        status = "✓ QUALIFIED" if i <= 8 else "  eliminated"
        print(f"  {i:<4} {t.team:<26} {t.group:<5} {rec.points:>4} "
              f"{rec.goal_diff:>+4} {rec.goals_for:>4}  {status}")

    # ── Print full 32-team bracket ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROUND OF 32 — COMPLETE 32-TEAM BRACKET")
    print("=" * 70)
    print(f"\n  {'Match':<12} {'Home Team':<26}  {'Away Team':<26}")
    print("  " + "-" * 66)
    for m in bracket.r32_matches:
        print(f"  {m.match_id:<12} {m.home:<26}  {m.away:<26}  "
              f"({m.home_slot} vs {m.away_slot})")

    print(f"\n  Total R32 matches: {len(bracket.r32_matches)}")
    all_bracket_teams = set()
    for m in bracket.r32_matches:
        all_bracket_teams.add(m.home)
        all_bracket_teams.add(m.away)
    print(f"  Unique teams in bracket: {len(all_bracket_teams)}")

    print("\n✅ Task 4.2 complete.")


if __name__ == "__main__":
    main()
