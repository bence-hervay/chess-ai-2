# Forward Chess — Rules

This file is the authoritative human-readable ruleset (plan §28). It
must be reviewed before any expensive Forward Chess training. The
implementation in `src/games/forward_chess.rs` and its rules corpus
follow this document exactly.

## 1. Board and coordinates

- The board is `width × height` (files `a…`, ranks `1…height`).
  Cell index = `(rank−1) · width + file_index`.
- White's home is rank 1; White's **forward** direction is toward
  higher ranks. Black's home is rank `height`; Black's forward is
  toward lower ranks.
- Four named rulesets (the exactness ladder):
  | Ruleset | Board | Pieces per side | Castling |
  |---|---|---|---|
  | `tiny` | 3×4 | K + P | no |
  | `small` | 4×4 | K + R + P | no |
  | `medium` | 6×6 | K + R + B + N + 3P | no |
  | `full` | 8×8 | standard chess array | yes |
  Initial layouts are listed in §12.

## 2. Orientation

- Every piece has an **orientation**: `Natural` or `Reversed`,
  defined **relative to its owner**. A `Natural` piece's *ahead*
  direction is its owner's forward; a `Reversed` piece's *ahead* is its
  owner's backward. Orientation is therefore separate from colour.
- All pieces start `Natural`. Only promotion creates `Reversed` pieces
  (§8). Orientation never changes after creation.

## 3. Core directional law (movement AND attack)

A piece may **move to** or **attack** a square only if the rank
displacement from its own square, measured in its *ahead* direction, is
**≥ 0**. Strictly-behind squares are neither reachable nor attacked.
Zero rank displacement (horizontal) is allowed wherever the ordinary
chess geometry of the piece allows horizontal movement.

The attack restriction is real, not only a movement restriction:
example (mandated by the plan): a Black `Natural` queen on `d2` does
**not** attack `d3` or `e3` (those are behind her), while she does
attack `d1`, `c1`, `e1`, and horizontally `a2…c2`, `e2…h2`.

## 4. Piece geometries (ordinary geometry filtered by §3)

With "ahead" = the piece's oriented forward:

- **King**: one step to the 5 squares that are ahead (3) or horizontal
  (2). Never the 3 behind squares.
- **Queen**: slides ahead, ahead-diagonally (2), and horizontally (2)
  — 5 rays.
- **Rook**: slides ahead and horizontally — 3 rays.
- **Bishop**: slides ahead-diagonally — 2 rays.
- **Knight**: the 4 jumps with positive ahead-displacement:
  (±1 file, +2 ahead) and (±2 file, +1 ahead).
- **Pawn** (always `Natural`; promotion replaces it): one step ahead to
  an empty square; from its home pawn rank (rank 2 for White, rank
  `height−1` for Black) optionally two steps ahead if both squares are
  empty **and the landing square is not the promotion rank**; captures
  one step diagonally ahead. Pawns have no horizontal move and never
  move without positive displacement.

Sliding pieces stop at the first occupied square (capture if enemy).
Pieces can become permanently immobile (e.g. a `Natural` knight on its
last rank); this is legal and such pieces still block squares but
attack nothing reachable.

## 5. Check, checkmate, stalemate

- A king is **in check** when an enemy piece attacks its square under
  §3–4 (a king may safely stand *behind* an enemy piece).
- A move is illegal if it leaves the mover's own king in check.
- No legal moves and in check: **checkmate** — the mover loses.
- No legal moves and not in check: **stalemate** — draw.

## 6. Castling (`full` ruleset only)

Castling is a purely **horizontal** king-and-rook move on the shared
home rank, and therefore compatible with §3:

- Conditions: neither the king nor the chosen rook has moved; every
  square between them is empty; the king is not in check; the two
  squares the king crosses (including the destination) are not attacked
  by any enemy piece (oriented attacks, §3).
- Effect: king moves two files toward the rook; the rook is placed on
  the square the king crossed. Both keep `Natural` orientation.

## 7. En passant

When a pawn advances two squares (§4) and lands beside an enemy pawn
on the same rank, that enemy pawn may — **immediately on the next move
only** — capture it as if it had advanced one square: the capturing
pawn moves diagonally ahead onto the crossed square and the double-step
pawn is removed. The en-passant state records the crossed square and
expires after one ply.

## 8. Promotion and orientation reversal

- A pawn reaching its **last rank** (rank `height` for White `Natural`
  pawns, rank 1 for Black `Natural` pawns) must promote to a knight,
  bishop, rook, or queen of its owner's colour.
- The promoted piece is created with **`Reversed` orientation**: its
  permitted movement and attack directions are the opposite of its
  owner's forward. Example: a White pawn promoting on `b8` to a rook
  yields a rook that slides toward lower ranks and horizontally, and
  never toward rank 8 again.

## 9. Repetition and move-count draws

- **Threefold repetition**: a position (piece placement with
  orientations, side to move, castling rights, en-passant state)
  occurring for the third time is an immediate draw.
- **Fifty-move rule**: 100 consecutive plies without a pawn move or a
  capture is an immediate draw.

## 10. Game termination summary

Win: checkmate the enemy king. Draw: stalemate, threefold repetition,
fifty-move rule. There is no other termination; the fifty-move rule
guarantees finiteness.

## 11. Raw feature encoding (no strategy features)

From the side to move's perspective (ranks mirrored for Black so
"forward" is always +rank): per occupied square, a single feature
`square × 24 + piece_type × 4 + (opponent? 2) + (reversed? 1)`;
then 4 castling-right features (own short/long, opponent short/long)
and `width` en-passant file features. Nothing else — no attack maps,
mobility, material, or positional conclusions.

Action IDs mirror ranks the same way: `from × cells + to` for ordinary
moves and queen promotions, plus `3 × width × 3` underpromotion slots
(piece × from-file × direction), exactly as in standard chess.

## 12. Initial layouts

- `tiny` (3×4): White `Kb1`, `Pa2`; Black `Kb4`, `Pc3`.
- `small` (4×4): White `Kb1`, `Rc1`, `Pa2`; Black mirrored
  (`Kc4`, `Rb4`, `Pd3`). (The 4×5 two-pawn variant measures at more
  than 60 million reachable positions — beyond even the 32 GB exact
  budget; see DECISIONS.md.)
- `medium` (6×6): White `Kc1`, `Rd1`, `Bb1`, `Ne1`, `Pa2`, `Pc2`,
  `Pf2`; Black mirrored on rank 6/5.
- `full` (8×8): the standard chess array (all pieces `Natural`).

Black's mirror = flip ranks and files so the position is rotationally
fair.
