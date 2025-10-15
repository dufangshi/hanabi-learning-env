## Hanabi Vector Encoding Cheat Sheet
When you reset the environment you get three return objects: (1) The agent's observations (`obs ) (2) the entire game state (`share_obs ) (3) Available actions(`available_actions)  

## (1) Agent Observation (obs)
The observation vector for each agent typically concatenates multiple encoded components:

| Segment | Description                           | Encoding Type         | Purpose                                  |
| ------- | ------------------------------------- | --------------------- | ---------------------------------------- |
| A       | Other players’ hands                  | One-hot per card      | Visible cards of teammates               |
| B       | Card knowledge (self-hand hints)      | Binary mask           | Which ranks/colors known about own cards |
| C       | Board state (fireworks, tokens, deck) | One-hot + thermometer | Game progress and available resources    |
| D       | Discard pile                          | Thermometer           | What cards are gone                      |
| E       | Recent moves                          | One-hot               | What the last player did                 |

These parts are flattened into a **single 1D vector** (often 300–400 elements in the 2-player setup).

#### **A. Other Players’ Hands**

> “Encodes cards in all other players’ hands (excluding own unknown hand).”
- **One-hot** per card using `num_colors × num_ranks` bits (e.g. 25 bits in standard Hanabi).
- Each card’s one-hot indicates exactly which colour and rank that card is.
- If a hand has 5 cards and there are 4 colors × 5 ranks → `5 × 25 = 125 bits` per player.
- If the deck is empty and a player’s hand has fewer cards, “missing” slots are encoded as zeros.
#### **B. Self-Hand Knowledge**
- Two boolean masks for each card:
    - Known colour(s) → `num_colors` bits
    - Known rank(s) → `num_ranks` bits
- These bits reflect clues received from teammates.
- Example:
    - You’ve been told your first card is “red” → red bit = 1, others = 0  
    - You’ve been told your third card is rank 2 → rank-2 bit = 1, others = 0    
#### **C. Board State**

| Feature                | Encoding                 | Description                               |
| ---------------------- | ------------------------ | ----------------------------------------- |
| **Fireworks**          | One-hot per color × rank | Which cards have been successfully played |
| **Information tokens** | Thermometer              | e.g., 000111 = 3 tokens left              |
| **Life tokens**        | Thermometer              | e.g., 110 = 2 lives left                  |
| **Deck size**          | Thermometer              | Remaining cards in deck                   |

Thermometer encoding = cumulative 1s up to current count (smooths value scale for neural nets).

---

#### **D. Discard Pile**

- Organized **colour-major**, same order as the game’s color string (“RYGWB”).
- For each color/rank:
    - 1 = at least one card of that rank discarded
    - More precisely, a **thermometer encoding** of how many copies of that card have been discarded.    
- Example (for a color):
    `LLL      H 1100011101`
    → 2 lowest ranks gone, none of next rank, 2 mid ranks gone, etc.
#### **E. Recent Action / Game Context**
- One-hot over all possible actions (`play/discard/hint`).
- Includes target player and hint type if relevant.
- Used to give the agent temporal awareness.
    

---


## (2) Hanabi Shared Observation Vector (`share_obs`) Cheat Sheet
`share_obs' is the **centralized input** to the **critic network** in MAPPO.  
It encodes **complete, global game state** — including **all agents’ hands**, **board**, **tokens**, **discards**, and **recent actions** — so the critic can estimate joint value functions.


| Segment                    | Encodes                     | Description                                                                                                     |
| -------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **0–(N × hand_size × 25)** | **All players’ hands**      | Each card = one-hot over 25 color–rank combinations. Includes **your own cards** (which are hidden from `obs`). |
| **Next few bits**          | **Missing card indicators** | Flags if a player’s hand has fewer than `hand_size` cards (e.g., deck is empty).                                |
| **Following section**      | **Board state**             | Fireworks, remaining deck size (thermometer), information tokens, and life tokens.                              |
| **Next section**           | **Discard pile**            | For each color, thermometer-encoded number of discarded cards per rank.                                         |
| **Next**                   | **Last action**             | One-hot encoded: which player acted, what type (play/discard/hint), and target.                                 |
| **Final section**          | **Turn and deck status**    | Encodes whose turn it is, deck size (thermometer), and any endgame flags.                                       |

### (3) Available actions(available_actions)  encoding
A binary mask indicating which actions are **legal** for each agent at the current timestep.
**Shape:** `[num_agents, action_dim]`

**Contains:**
- `1` → Action is allowed
- `0` → Action is illegal (e.g., no info tokens left to hint, or cannot play a missing card)
**Example:**
`available_actions[0] = [1, 1, 0, 0, 1, 0] # Agent 0 can take actions 0, 1, and 4 only`

The actions are **enumerated** and include:
1. **Play actions** — one per card in the hand.
2. **Discard actions** — one per card in the hand.
3. **Hint actions** — one for each valid combination of `(target player × color)` and `(target player × rank)`.
For a standard 2-player, 5-card game:
- 5 play actions
- 5 discard actions
- 2 × (5 colors + 5 ranks) = 20 hint actions
- Total: **30 discrete actions**
So `available_actions` would be length 30.


Example:
```
`available_actions[0] = [1, 1, 0, 0, 1,  # play/discard actions     
						0, 0, 0, 0, 0,  # hint Player 1 by color
						1, 1, 1, 1, 1,  # hint Player 1 by rank
						0, 0, 0, 0, 0,  # hint Player 2 by color
						1, 0, 0, 0, 0   # hint Player 2 by rank ]`
```

Interpretation:
- You can **play** cards 0, 1, and 4
- You can **hint ranks** 0–4 to Player 1
- You can **hint rank 0** to Player 2
- You cannot discard (maybe no discards allowed or wrong phase)
- You cannot hint colors (maybe out of info tokens)