"""
Gameplay trace logging for detailed Hanabi game analysis.

This module provides functionality to capture detailed turn-by-turn gameplay traces,
including state observations, actions, and results.
"""

import sys
from pathlib import Path

# Add pyhanabi to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "hanabi"))

import pyhanabi


class GameplayTraceLogger:
    """
    Logs detailed gameplay traces for Hanabi games.

    Captures turn-by-turn state, actions, and results for debugging and analysis.
    """

    def __init__(self, num_players: int):
        """
        Initialize the trace logger.

        Args:
            num_players: Number of players in the game
        """
        self.num_players = num_players
        self.trace_lines = []
        self.turn_number = 0

        # Color index to character mapping
        self.color_chars = ['R', 'Y', 'G', 'W', 'B']

    def log_turn_start(self, player_id: int, observation: dict, state):
        """
        Log the state at the start of a turn before action is taken.

        Args:
            player_id: ID of the player taking action
            observation: Full observation dict from _make_observation_all_players()
            state: HanabiState object
        """
        self.turn_number += 1

        self.trace_lines.append(f"\n{'='*60}")
        self.trace_lines.append(f"====round {self.turn_number}====")
        self.trace_lines.append(f"{'='*60}")
        self.trace_lines.append(f"Player {player_id}'s turn")
        self.trace_lines.append("")

        # Get player's observation
        player_obs = observation['player_observations'][player_id]

        # State information
        self.trace_lines.append("State:")

        # Fireworks
        fireworks = player_obs['fireworks']
        fw_str = " ".join([f"{color}:{level}" for color, level in fireworks.items()])
        self.trace_lines.append(f"  Fireworks: {fw_str}")

        # Tokens
        info_tokens = player_obs['information_tokens']
        life_tokens = player_obs['life_tokens']
        deck_size = player_obs['deck_size']
        self.trace_lines.append(f"  Tokens: Info={info_tokens}/8, Life={life_tokens}/3, Deck={deck_size}")

        # Own hand knowledge
        own_knowledge = player_obs['card_knowledge'][0]  # First element is own hand
        hand_str = self._format_hand_knowledge(own_knowledge)
        self.trace_lines.append(f"  Own hand knowledge: {hand_str}")

        # Partner hands (visible to this player)
        for offset in range(1, self.num_players):
            partner_idx = (player_id + offset) % self.num_players
            partner_hand = player_obs['observed_hands'][offset]
            partner_str = self._format_visible_hand(partner_hand)
            self.trace_lines.append(f"  Player {partner_idx} hand (visible): {partner_str}")

        # Recent discards (last 5)
        discard_pile = player_obs['discard_pile']
        if discard_pile:
            recent = discard_pile[-5:] if len(discard_pile) > 5 else discard_pile
            discard_str = self._format_visible_hand(recent)
            self.trace_lines.append(f"  Recent discards: {discard_str} (total: {len(discard_pile)})")
        else:
            self.trace_lines.append(f"  Recent discards: [] (total: 0)")

        self.trace_lines.append("")

    def log_action(self, player_id: int, action_dict: dict):
        """
        Log the action taken by the player.

        Args:
            player_id: ID of the player taking action
            action_dict: Action dictionary with 'action_type' and other fields
        """
        action_type = action_dict['action_type']

        if action_type == 'PLAY':
            card_idx = action_dict['card_index']
            action_str = f"PLAY card {card_idx}"
        elif action_type == 'DISCARD':
            card_idx = action_dict['card_index']
            action_str = f"DISCARD card {card_idx}"
        elif action_type == 'REVEAL_COLOR':
            color = action_dict['color']
            target = action_dict['target_offset']
            action_str = f"REVEAL_COLOR {color} to player +{target}"
        elif action_type == 'REVEAL_RANK':
            rank = action_dict['rank']
            target = action_dict['target_offset']
            action_str = f"REVEAL_RANK {rank+1} to player +{target}"
        else:
            action_str = f"{action_type}"

        self.trace_lines.append(f"Action: {action_str}")

    def log_action_result(self, observation: dict, reward: float, info: dict):
        """
        Log the result of the action after environment step.

        Args:
            observation: New observation after step
            reward: Reward received
            info: Info dict containing score
        """
        # Get the most recent move from history to determine result
        # We can infer from reward and score change

        score = info.get('score', 0)

        self.trace_lines.append("")
        self.trace_lines.append("Result:")

        # Interpret reward and score
        if reward > 0:
            self.trace_lines.append(f"  SUCCESS! Score change: +{int(reward)}, New score: {score}")
        elif reward < -10:
            # Death penalty
            self.trace_lines.append(f"  MISPLAY! Life tokens lost. Penalty: {reward}, Score: {score}")
        elif reward < 0:
            # Regular misplay
            self.trace_lines.append(f"  MISPLAY! Score: {score}")
        else:
            # Discard or hint (no score change)
            self.trace_lines.append(f"  Action completed. Score: {score}")

        # Show token changes
        player_obs = observation['player_observations'][0]
        info_tokens = player_obs['information_tokens']
        life_tokens = player_obs['life_tokens']
        self.trace_lines.append(f"  Tokens: Info={info_tokens}/8, Life={life_tokens}/3")

    def log_game_end(self, state, num_turns: int):
        """
        Log final game summary.

        Args:
            state: Final HanabiState
            num_turns: Total number of turns played
        """
        self.trace_lines.append(f"\n{'='*60}")
        self.trace_lines.append("=== GAME END ===")
        self.trace_lines.append(f"{'='*60}")
        self.trace_lines.append("")

        final_score = state.score()
        life_tokens = state.life_tokens()

        self.trace_lines.append(f"Final Score: {final_score}/25")
        self.trace_lines.append(f"Total Turns: {num_turns}")
        self.trace_lines.append(f"Life Tokens Remaining: {life_tokens}/3")

        # Determine end reason
        if life_tokens == 0:
            end_reason = "BOMBED_OUT (life tokens exhausted)"
        elif state.deck_size() == 0:
            end_reason = "OUT_OF_CARDS (deck exhausted)"
        else:
            end_reason = "GAME_COMPLETE"

        self.trace_lines.append(f"End Reason: {end_reason}")

        # Final fireworks state
        fireworks = state.fireworks()
        fw_str = " ".join([f"{self.color_chars[i]}:{fireworks[i]}" for i in range(5)])
        self.trace_lines.append(f"Final Fireworks: {fw_str}")

        # Discard pile summary
        discard_pile = state.discard_pile()
        self.trace_lines.append(f"Cards Discarded: {len(discard_pile)}")

        self.trace_lines.append("")

    def get_trace(self) -> str:
        """
        Get the complete trace as a string.

        Returns:
            Complete trace log as string
        """
        return "\n".join(self.trace_lines)

    def _format_hand_knowledge(self, knowledge_list: list) -> str:
        """
        Format hand knowledge for display.

        Args:
            knowledge_list: List of card knowledge dicts

        Returns:
            Formatted string like "[R/?, ?/2, R/1, ?/?, W/?]"
        """
        cards = []
        for k in knowledge_list:
            color = k['color'] if k['color'] is not None else '?'
            rank = str(k['rank'] + 1) if k['rank'] is not None else '?'
            cards.append(f"{color}/{rank}")

        return "[" + ", ".join(cards) + "]"

    def _format_visible_hand(self, hand_list: list) -> str:
        """
        Format visible hand for display.

        Args:
            hand_list: List of card dicts with 'color' and 'rank'

        Returns:
            Formatted string like "[R/1, Y/2, G/1, W/3, B/1]"
        """
        cards = []
        for card in hand_list:
            if card['color'] is None or card['rank'] == -1:
                cards.append("?/?")
            else:
                color = card['color']
                rank = card['rank'] + 1
                cards.append(f"{color}/{rank}")

        return "[" + ", ".join(cards) + "]"
