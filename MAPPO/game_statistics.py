"""
Game Statistics Tracker for Hanabi MAPPO Evaluation

This module provides detailed analysis of agent behavior during gameplay,
including play decisions, discard patterns, and hint effectiveness.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

# Add pyhanabi to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "hanabi"))

try:
    import pyhanabi
    from pyhanabi import HanabiMoveType
except ImportError:
    pyhanabi = None
    HanabiMoveType = None


class PlayDecisionType(Enum):
    """Classification of play decisions based on agent's knowledge"""
    FULLY_KNOWN_PLAYABLE = "fully_known_playable"      # Agent knows both color and rank, card is playable
    PARTIALLY_KNOWN_PLAYABLE = "partially_known_playable"  # Agent can infer playability from hints
    CONVENTION_PLAY = "convention_play"                # Agent plays without complete info (hidden conventions)
    LUCKY_BLIND_PLAY = "lucky_blind_play"             # No hints, but card happens to be playable
    MISPLAY = "misplay"                                # Card is not playable


class DiscardDecisionType(Enum):
    """Classification of discard decisions"""
    KNOWN_USEFUL_DISCARD = "known_useful_discard"      # Agent discards a card they know is still needed
    KNOWN_USELESS_DISCARD = "known_useless_discard"    # Agent correctly discards a useless card
    UNKNOWN_DISCARD = "unknown_discard"                # Agent discards without full knowledge


@dataclass
class PlayStatistics:
    """Statistics for play decisions"""
    fully_known_playable: int = 0
    partially_known_playable: int = 0
    convention_play: int = 0
    lucky_blind_play: int = 0
    misplay: int = 0

    @property
    def total(self) -> int:
        return (self.fully_known_playable + self.partially_known_playable +
                self.convention_play + self.lucky_blind_play + self.misplay)

    @property
    def successful_plays(self) -> int:
        return self.total - self.misplay

    @property
    def success_rate(self) -> float:
        return self.successful_plays / self.total if self.total > 0 else 0.0


@dataclass
class DiscardStatistics:
    """Statistics for discard decisions"""
    known_useful_discard: int = 0
    known_useless_discard: int = 0
    unknown_discard: int = 0

    @property
    def total(self) -> int:
        return self.known_useful_discard + self.known_useless_discard + self.unknown_discard

    @property
    def correct_discards(self) -> int:
        return self.known_useless_discard


@dataclass
class HintStatistics:
    """Statistics for hint effectiveness"""
    total_hints: int = 0
    color_hints: int = 0
    rank_hints: int = 0
    total_cards_revealed: int = 0
    total_new_info: int = 0

    @property
    def avg_cards_revealed(self) -> float:
        return self.total_cards_revealed / self.total_hints if self.total_hints > 0 else 0.0

    @property
    def avg_new_info(self) -> float:
        return self.total_new_info / self.total_hints if self.total_hints > 0 else 0.0

    @property
    def hint_efficiency(self) -> float:
        """Ratio of new information to total cards revealed"""
        return self.total_new_info / self.total_cards_revealed if self.total_cards_revealed > 0 else 0.0


@dataclass
class GameStatistics:
    """Complete statistics for a single game"""
    play_stats: PlayStatistics = field(default_factory=PlayStatistics)
    discard_stats: DiscardStatistics = field(default_factory=DiscardStatistics)
    hint_stats: HintStatistics = field(default_factory=HintStatistics)

    final_score: int = 0
    life_tokens_remaining: int = 0
    turns_taken: int = 0


class GameStatisticsTracker:
    """
    Analyzes a completed Hanabi game and generates detailed statistics.
    """

    def __init__(self, state, game, num_players: int):
        """
        Initialize tracker with a completed game state.

        Args:
            state: HanabiState object from pyhanabi
            game: HanabiGame object (needed to query card counts)
            num_players: Number of players in the game
        """
        self.state = state
        self.game = game
        self.num_players = num_players
        self.stats = GameStatistics()

    def analyze(self, debug: bool = False) -> GameStatistics:
        """
        Analyze the complete game and return statistics.

        Args:
            debug: If True, print detailed debugging information

        Returns:
            GameStatistics object with detailed analysis
        """
        if pyhanabi is None:
            raise ImportError("pyhanabi is not available")

        if debug:
            print("[DEBUG] Starting game statistics analysis")
            print(f"[DEBUG] State is_terminal: {self.state.is_terminal()}")
            print(f"[DEBUG] State life_tokens: {self.state.life_tokens()}")
            print(f"[DEBUG] State score: {self.state.score()}")

        # Get move history
        if debug:
            print("[DEBUG] Calling state.move_history()...")
        history = self.state.move_history()
        if debug:
            print(f"[DEBUG] Got {len(history)} history items")

        # Analyze each move
        for idx, hist_item in enumerate(history):
            if debug:
                print(f"[DEBUG] Processing history item {idx+1}/{len(history)}")

            if debug:
                print(f"[DEBUG]   Calling hist_item.move()...")
            move = hist_item.move()
            if debug:
                print(f"[DEBUG]   Calling hist_item.player()...")
            player_id = hist_item.player()
            if debug:
                print(f"[DEBUG]   Calling move.type()...")
            move_type = move.type()
            if debug:
                print(f"[DEBUG]   Move type: {move_type}, Player: {player_id}")

            if move_type == HanabiMoveType.PLAY:
                if debug:
                    print(f"[DEBUG]   Analyzing PLAY move...")
                self._analyze_play(hist_item, player_id, debug=debug)
            elif move_type == HanabiMoveType.DISCARD:
                if debug:
                    print(f"[DEBUG]   Analyzing DISCARD move...")
                self._analyze_discard(hist_item, player_id, debug=debug)
            elif move_type in [HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK]:
                if debug:
                    print(f"[DEBUG]   Analyzing HINT move...")
                self._analyze_hint(hist_item, debug=debug)

        # Record final state
        if debug:
            print("[DEBUG] Recording final state...")
            print("[DEBUG]   Calling state.score()...")
        self.stats.final_score = self.state.score()
        if debug:
            print("[DEBUG]   Calling state.life_tokens()...")
        self.stats.life_tokens_remaining = self.state.life_tokens()
        if debug:
            print("[DEBUG]   Counting turns...")
        self.stats.turns_taken = len([h for h in history if h.move().type() != HanabiMoveType.DEAL])

        if debug:
            print("[DEBUG] Analysis complete!")

        return self.stats

    def _analyze_play(self, hist_item, player_id: int, debug: bool = False):
        """
        Analyze a PLAY move and classify it.

        Args:
            hist_item: HanabiHistoryItem for this move
            player_id: Player who made the move
            debug: If True, print debugging information
        """
        if debug:
            print("[DEBUG]     Calling move.card_index()...")
        move = hist_item.move()
        card_index = move.card_index()

        # Get the actual card that was played
        if debug:
            print("[DEBUG]     Calling hist_item.color()...")
        actual_color = hist_item.color()
        if debug:
            print("[DEBUG]     Calling hist_item.rank()...")
        actual_rank = hist_item.rank()
        if debug:
            print("[DEBUG]     Calling hist_item.scored()...")
        was_successful = hist_item.scored()
        if debug:
            print(f"[DEBUG]     Card: color={actual_color}, rank={actual_rank}, scored={was_successful}")

        # Reconstruct what the player knew at the time of the play
        # We need to look at the observation BEFORE this move
        if debug:
            print("[DEBUG]     Getting card knowledge...")
        knowledge = self._get_card_knowledge_before_move(player_id, card_index, hist_item, debug=debug)

        if knowledge is None:
            # Fallback: classify based on success
            if debug:
                print(f"[DEBUG]     No knowledge available, classifying by success: {was_successful}")
            if was_successful:
                self.stats.play_stats.convention_play += 1
            else:
                self.stats.play_stats.misplay += 1
            return

        # Classify the play decision
        color_known = knowledge.color() is not None
        rank_known = knowledge.rank() is not None

        if not was_successful:
            # Misplay
            self.stats.play_stats.misplay += 1
        elif color_known and rank_known:
            # Fully known playable
            self.stats.play_stats.fully_known_playable += 1
        elif self._is_inferable(knowledge):
            # Can infer full identity through elimination
            self.stats.play_stats.partially_known_playable += 1
        elif color_known or rank_known:
            # Has some info (convention play)
            self.stats.play_stats.convention_play += 1
        else:
            # No hints at all (lucky blind play)
            self.stats.play_stats.lucky_blind_play += 1

    def _analyze_discard(self, hist_item, player_id: int, debug: bool = False):
        """
        Analyze a DISCARD move and classify it.

        Args:
            hist_item: HanabiHistoryItem for this move
            player_id: Player who made the move
            debug: If True, print debugging information
        """
        if debug:
            print("[DEBUG]     Calling move.card_index()...")
        move = hist_item.move()
        card_index = move.card_index()

        # Get the actual card that was discarded
        if debug:
            print("[DEBUG]     Calling hist_item.color()...")
        actual_color = hist_item.color()
        if debug:
            print("[DEBUG]     Calling hist_item.rank()...")
        actual_rank = hist_item.rank()
        if debug:
            print(f"[DEBUG]     Card: color={actual_color}, rank={actual_rank}")

        # Get what the player knew
        if debug:
            print("[DEBUG]     Getting card knowledge...")
        knowledge = self._get_card_knowledge_before_move(player_id, card_index, hist_item, debug=debug)

        if knowledge is None:
            if debug:
                print("[DEBUG]     No knowledge available")
            self.stats.discard_stats.unknown_discard += 1
            return

        # Check if player had full knowledge
        color_known = knowledge.color() is not None
        rank_known = knowledge.rank() is not None

        if not (color_known and rank_known):
            # Player didn't have full knowledge
            self.stats.discard_stats.unknown_discard += 1
            return

        # Player knew the card - check if it was useful
        if debug:
            print(f"[DEBUG]     Checking if card is useful (color={actual_color}, rank={actual_rank})...")
        if self._is_card_useful(actual_color, actual_rank, debug=debug):
            # Discarded a useful card (bad decision)
            self.stats.discard_stats.known_useful_discard += 1
        else:
            # Discarded a useless card (good decision)
            self.stats.discard_stats.known_useless_discard += 1

    def _analyze_hint(self, hist_item, debug: bool = False):
        """
        Analyze a REVEAL (hint) move.

        Args:
            hist_item: HanabiHistoryItem for this move
            debug: If True, print debugging information
        """
        if debug:
            print("[DEBUG]     Getting move type...")
        move = hist_item.move()

        self.stats.hint_stats.total_hints += 1

        if move.type() == HanabiMoveType.REVEAL_COLOR:
            self.stats.hint_stats.color_hints += 1
        else:
            self.stats.hint_stats.rank_hints += 1

        # Get hint effectiveness
        if debug:
            print("[DEBUG]     Calling hist_item.card_info_revealed()...")
        revealed_cards = hist_item.card_info_revealed()
        if debug:
            print("[DEBUG]     Calling hist_item.card_info_newly_revealed()...")
        newly_revealed = hist_item.card_info_newly_revealed()

        self.stats.hint_stats.total_cards_revealed += len(revealed_cards)
        self.stats.hint_stats.total_new_info += len(newly_revealed)

    def _get_card_knowledge_before_move(self, player_id: int, card_index: int,
                                       current_hist_item, debug: bool = False) -> Optional[object]:
        """
        Reconstruct what the player knew about a card before making a move.

        This is complex because we need to replay hints to understand
        the knowledge state at a specific point in time.

        For now, we use a simplified approach that avoids accessing terminated states.

        Args:
            player_id: Player whose knowledge we're checking
            card_index: Index of the card in the player's hand
            current_hist_item: The history item of the current move
            debug: If True, print debugging information

        Returns:
            CardKnowledge object or None if unavailable
        """
        # IMPORTANT: Avoid calling state.observation() on terminated states
        # as it may trigger assertion failures in the C++ layer
        # For now, we return None and rely on heuristics
        # TODO: Implement proper game replay to accurately track card knowledge
        if debug:
            print("[DEBUG]       Returning None (knowledge tracking not implemented)")
        return None

    def _is_inferable(self, knowledge) -> bool:
        """
        Check if a card's identity can be inferred from plausibility.

        Args:
            knowledge: CardKnowledge object

        Returns:
            True if card identity can be uniquely determined
        """
        # Count plausible colors and ranks
        plausible_colors = sum(1 for c in range(5) if knowledge.color_plausible(c))
        plausible_ranks = sum(1 for r in range(5) if knowledge.rank_plausible(r))

        # Card is inferable if only one possibility remains for each
        return plausible_colors == 1 and plausible_ranks == 1

    def _is_card_useful(self, color: int, rank: int, debug: bool = False) -> bool:
        """
        Determine if a card is still useful for the game.

        A card is useful if:
        1. It hasn't been played yet (rank >= fireworks[color])
        2. Not all copies of it have been discarded

        Args:
            color: Card color (0-4)
            rank: Card rank (0-4)
            debug: If True, print debugging information

        Returns:
            True if the card is still useful
        """
        # Check if already played
        if debug:
            print(f"[DEBUG]       Calling state.fireworks()...")
        fireworks = self.state.fireworks()
        if debug:
            print(f"[DEBUG]       Fireworks state: {fireworks}")

        if rank < fireworks[color]:
            if debug:
                print(f"[DEBUG]       Card already played (rank {rank} < fireworks[{color}]={fireworks[color]})")
            return False

        # Count how many of this card are in discard pile
        if debug:
            print(f"[DEBUG]       Calling state.discard_pile()...")
        discard_pile = self.state.discard_pile()
        if debug:
            print(f"[DEBUG]       Discard pile has {len(discard_pile)} cards")

        discard_count = sum(
            1 for card in discard_pile
            if card.color() == color and card.rank() == rank
        )
        if debug:
            print(f"[DEBUG]       Found {discard_count} copies in discard pile")

        # Get total number of this card type in the game
        if debug:
            print(f"[DEBUG]       Calling game.num_cards({color}, {rank})...")
        total_count = self.game.num_cards(color, rank)
        if debug:
            print(f"[DEBUG]       Total count: {total_count}")

        # Card is useful if not all copies are discarded
        is_useful = discard_count < total_count
        if debug:
            print(f"[DEBUG]       Card is {'useful' if is_useful else 'NOT useful'}")
        return is_useful


class GameStatisticsCollector:
    """
    Collects statistics across multiple games during evaluation.
    """

    def __init__(self):
        self.episode_stats: List[GameStatistics] = []

    def add_episode(self, stats: GameStatistics):
        """Add statistics from a completed episode."""
        self.episode_stats.append(stats)

    def get_aggregate_statistics(self) -> Dict:
        """
        Compute aggregate statistics across all episodes.

        Returns:
            Dictionary with aggregated statistics
        """
        if not self.episode_stats:
            return {}

        # Aggregate play statistics
        total_play_stats = PlayStatistics()
        for stats in self.episode_stats:
            total_play_stats.fully_known_playable += stats.play_stats.fully_known_playable
            total_play_stats.partially_known_playable += stats.play_stats.partially_known_playable
            total_play_stats.convention_play += stats.play_stats.convention_play
            total_play_stats.lucky_blind_play += stats.play_stats.lucky_blind_play
            total_play_stats.misplay += stats.play_stats.misplay

        # Aggregate discard statistics
        total_discard_stats = DiscardStatistics()
        for stats in self.episode_stats:
            total_discard_stats.known_useful_discard += stats.discard_stats.known_useful_discard
            total_discard_stats.known_useless_discard += stats.discard_stats.known_useless_discard
            total_discard_stats.unknown_discard += stats.discard_stats.unknown_discard

        # Aggregate hint statistics
        total_hint_stats = HintStatistics()
        for stats in self.episode_stats:
            total_hint_stats.total_hints += stats.hint_stats.total_hints
            total_hint_stats.color_hints += stats.hint_stats.color_hints
            total_hint_stats.rank_hints += stats.hint_stats.rank_hints
            total_hint_stats.total_cards_revealed += stats.hint_stats.total_cards_revealed
            total_hint_stats.total_new_info += stats.hint_stats.total_new_info

        # Compute averages
        num_episodes = len(self.episode_stats)
        avg_score = sum(s.final_score for s in self.episode_stats) / num_episodes
        avg_turns = sum(s.turns_taken for s in self.episode_stats) / num_episodes

        return {
            'num_episodes': num_episodes,
            'avg_score': avg_score,
            'avg_turns': avg_turns,
            'play_stats': total_play_stats,
            'discard_stats': total_discard_stats,
            'hint_stats': total_hint_stats,
        }

    def generate_report(self) -> str:
        """
        Generate a human-readable statistics report.

        Returns:
            Formatted string report
        """
        agg = self.get_aggregate_statistics()

        if not agg:
            return "No statistics available."

        play_stats = agg['play_stats']
        discard_stats = agg['discard_stats']
        hint_stats = agg['hint_stats']

        lines = [
            "=" * 80,
            "GAME STATISTICS REPORT",
            "=" * 80,
            f"Episodes analyzed: {agg['num_episodes']}",
            f"Average score: {agg['avg_score']:.2f}",
            f"Average turns: {agg['avg_turns']:.1f}",
            "",
            f"PLAY DECISIONS (total: {play_stats.total:,}):",
            f"  Fully Known Playable:    {play_stats.fully_known_playable:,} ({100*play_stats.fully_known_playable/play_stats.total if play_stats.total > 0 else 0:.1f}%) ✓",
            f"  Partially Known Playable: {play_stats.partially_known_playable:,} ({100*play_stats.partially_known_playable/play_stats.total if play_stats.total > 0 else 0:.1f}%) ✓",
            f"  Convention Play:          {play_stats.convention_play:,} ({100*play_stats.convention_play/play_stats.total if play_stats.total > 0 else 0:.1f}%) ~",
            f"  Lucky Blind Play:         {play_stats.lucky_blind_play:,} ({100*play_stats.lucky_blind_play/play_stats.total if play_stats.total > 0 else 0:.1f}%)",
            f"  Misplay:                  {play_stats.misplay:,} ({100*play_stats.misplay/play_stats.total if play_stats.total > 0 else 0:.1f}%) ✗",
            f"  Success Rate:             {100*play_stats.success_rate:.1f}%",
            "",
            f"DISCARD DECISIONS (total: {discard_stats.total:,}):",
            f"  Known Useful Discard:    {discard_stats.known_useful_discard:,} ({100*discard_stats.known_useful_discard/discard_stats.total if discard_stats.total > 0 else 0:.1f}%) ✗ [CRITICAL]",
            f"  Known Useless Discard:   {discard_stats.known_useless_discard:,} ({100*discard_stats.known_useless_discard/discard_stats.total if discard_stats.total > 0 else 0:.1f}%) ✓",
            f"  Unknown Discard:         {discard_stats.unknown_discard:,} ({100*discard_stats.unknown_discard/discard_stats.total if discard_stats.total > 0 else 0:.1f}%)",
            "",
            f"HINT EFFECTIVENESS (total: {hint_stats.total_hints:,}):",
            f"  Color hints: {hint_stats.color_hints:,} ({100*hint_stats.color_hints/hint_stats.total_hints if hint_stats.total_hints > 0 else 0:.1f}%)",
            f"  Rank hints:  {hint_stats.rank_hints:,} ({100*hint_stats.rank_hints/hint_stats.total_hints if hint_stats.total_hints > 0 else 0:.1f}%)",
            f"  Avg cards revealed per hint: {hint_stats.avg_cards_revealed:.2f}",
            f"  Avg new info per hint:       {hint_stats.avg_new_info:.2f}",
            f"  Hint efficiency:             {100*hint_stats.hint_efficiency:.1f}%",
            "=" * 80,
        ]

        return "\n".join(lines)
