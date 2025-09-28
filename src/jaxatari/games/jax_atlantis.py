import os
from dataclasses import dataclass, field

from jax import config, Array

import jax.lax
import jax.numpy as jnp
import chex

import pygame
from typing import Dict, Any, Optional, NamedTuple, Tuple
from functools import partial

from jaxatari.rendering import jax_rendering_utils as aj
from jaxatari.renderers import JAXGameRenderer
from jaxatari.environment import (
    JaxEnvironment,
    JAXAtariAction as Action,
    EnvObs,
)
import jaxatari.spaces as spaces


@dataclass(frozen=True)
class GameConfig:
    """Game configuration parameters"""

    screen_width: int = 160
    screen_height: int = 210
    scaling_factor: int = 3
    bullet_height: int = 1
    bullet_width: int = 1
    bullet_speed: int = 3  # for side cannon 3 in x 2 in y middle 3 in y
    cannon_height: int = 8
    cannon_width: int = 8
    cannon_y: jnp.ndarray = field(
        default_factory=lambda: jnp.array([118, 106, 106], dtype=jnp.int32)
    )
    cannon_x: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0, 72, 152], dtype=jnp.int32)
    )
    max_bullets: int = 2  # Max 2 bullets simultaneously at screen
    max_enemies: int = 20  # max 1 enemy per lane
    fire_cooldown_frames: int = 9  # delay between shots
    # y-coordinates of the different enemy paths/heights
    enemy_paths: jnp.ndarray = field(
        default_factory=lambda: jnp.array([20, 40, 60, 80], dtype=jnp.int32)
    )
    enemy_probabilities: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.45, 0.45, 0.10], dtype=jnp.float32)
    )
    enemy_points: jnp.ndarray = field(
        default_factory=lambda: jnp.array([100, 100, 1000], dtype=jnp.int32)
    )
    enemy_speed_multipliers: jnp.ndarray = field(
        default_factory=lambda: jnp.array([1, 1, 2], dtype=jnp.int32)
    )
    enemy_width: jnp.ndarray = field(
        default_factory=lambda: jnp.array([15, 15, 9], dtype=jnp.int32)
    )
    enemy_height: jnp.ndarray = field(
        default_factory=lambda: jnp.array([7, 8, 7], dtype=jnp.int32)
    )
    # enemy_width: int = 15  # 3 different lengths 15, 16, 9
    # enemy_height: int = 8
    enemy_speed: int = 1  # changes throughout the game
    enemy_spawn_min_frames: int = 5
    enemy_spawn_max_frames: int = 50
    wave_end_cooldown: int = (
        150  # cooldown of 150 frames after wave-end, before spawning new enemies
    )
    wave_start_enemy_count: int = 10  # number of enemies in the first wave
    max_digits_for_score: int = (
        9  # highest possible score has length of 9; lower limit is always possible
    )
    # coordinates and sizes of all installations
    installations_y: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [164, 142, 132, 118, 153, 132], dtype=jnp.int32
        )
    )
    installations_x: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [17, 38, 62, 82, 96, 142], dtype=jnp.int32
        )
    )
    installations_width: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [16, 16, 4, 4, 16, 4], dtype=jnp.int32
        )
    )
    installations_height: int = 8  # all the same height
    height_upper_beam = 40
    start_beam = 90


# Each value of this class is a list.
# e.g. if i have 3 entities, then each of these lists would have a length of 3
class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    alive: jnp.ndarray


class AtlantisState(NamedTuple):
    score: chex.Array  # tracks the current score
    reward: chex.Array
    score_spent: chex.Array  # tracks how much was spent on repair
    wave: chex.Array  # tracks which wave we are in

    # columns = [ x,  y,  dx,   type_id, lane, active_flag ]
    #   x, y        → position
    #   dx          → horizontal speed (positive or negative)
    #   type_id     → integer index into enemy_specs dict
    #   lane        → current lane the enemy is on
    #   active_flag → 1 if on-screen, 0 otherwise
    enemies: chex.Array  # shape: (max_enemies, 6)

    # columns = [ x, y, dx, dy]. dx and dy is the velocity
    bullets: chex.Array  # shape: (max_bullets, 4)
    bullets_alive: chex.Array  # stores all the active bullets as bools
    fire_cooldown: chex.Array  # frames left until next shot
    fire_button_prev: chex.Array  # was fire button down last frame
    enemy_spawn_timer: chex.Array  # frames until next spawn
    rng: chex.Array  # PRNG state
    lanes_free: chex.Array  # bool for each lane
    command_post_alive: chex.Array  # is command post alive (middle cannon)
    number_enemies_wave_remaining: (
        chex.Array
    )  # number of remaining enemies per wave
    wave_end_cooldown_remaining: chex.Array
    installations: chex.Array  # stores boolean alive
    # Plasma is deactivated once a cannon or an installment is hit, after that the enemy
    # should reach the end of the screen until it is reactivated. See _refresh_plasma_active for more info.
    plasma_active: chex.Array


class AtlantisObservation(NamedTuple):
    score: jnp.ndarray
    enemy: EntityPosition
    bullet: EntityPosition
    installations_alive: jnp.ndarray
    command_post_alive: jnp.ndarray


class AtlantisInfo(NamedTuple):
    score: jnp.ndarray
    wave: jnp.ndarray
    enemies_alive: jnp.ndarray  # scalar
    bullets_alive: jnp.ndarray  # scalar
    enemies_remaining_in_wave: jnp.ndarray
    wave_cooldown_remaining: jnp.ndarray
    command_post_alive: jnp.ndarray  # bool scalar
    installations_alive: jnp.ndarray  # (6,) bool
    all_rewards: jnp.ndarray


class AtlantisConstants(NamedTuple):
    pass


class Renderer_AtraJaxis(JAXGameRenderer):
    sprites: Dict[str, Any]

    def __init__(self, config: GameConfig | None = None):
        super().__init__()
        self.config = config or GameConfig()
        self.sprite_path = (
            f"{os.path.dirname(os.path.abspath(__file__))}/sprites/atlantis"
        )
        self.sprites = self._load_sprites()
        self.score_digit_sprites = self.sprites.get("score_digit_sprites")

    def _load_sprites(self) -> dict[str, Any]:
        """Loads all necessary sprites from .npy files."""
        sprites: Dict[str, Any] = {}

        # Helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f"{name}.npy")
            frame = aj.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)

        # Load Sprites
        # Backgrounds + Dynamic elements + UI elements
        sprite_names = [
            "small_enemy",
            "round_enemy",
            "long_enemy",
            "cannon_left",
            "cannon_right",
            "cannon_middle",
            "installation_1",
            "installation_2",
            "installation_3",
            "installation_4",
            "installation_5",
            "installation_6",
            "background",
        ]
        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                sprites[name] = loaded_sprite

        # digits for score
        sprites["score_digit_sprites"] = aj.load_and_pad_digits(
            os.path.join(self.sprite_path, "score_{}.npy"), num_chars=10
        )
        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: AtlantisState) -> chex.Array:

        def _solid_sprite(
            width: int, height: int, rgb: tuple[int, int, int]
        ) -> chex.Array:
            """Creates a slid-color RGBA sprite of given size and color"""
            rgb_arr = jnp.broadcast_to(
                jnp.array(rgb, dtype=jnp.uint8), (width, height, 3)
            )
            alpha = jnp.full((width, height, 1), 255, dtype=jnp.uint8)
            return jnp.concatenate([rgb_arr, alpha], axis=-1)  # (W, H, 4)

        cfg = self.config
        W, H = cfg.screen_width, cfg.screen_height

        # add background
        bg_sprite = self.sprites["background"]
        raster = aj.render_at(
            jnp.zeros_like(bg_sprite[..., :3]), 0, 0, bg_sprite
        )

        # add cannons
        raster = aj.render_at(
            raster,
            cfg.cannon_x[0],
            cfg.cannon_y[0],
            self.sprites["cannon_left"],
        )
        raster = jax.lax.cond(
            state.command_post_alive,
            lambda r: aj.render_at(
                r,
                cfg.cannon_x[1],
                cfg.cannon_y[1],
                self.sprites["cannon_middle"],
            ),
            lambda r: r,  # return raster unchanged if false
            raster,
        )
        raster = aj.render_at(
            raster,
            cfg.cannon_x[2],
            cfg.cannon_y[2],
            self.sprites["cannon_right"],
        )

        # add installations
        raster = jax.lax.cond(
            state.installations[0],
            lambda r: aj.render_at(
                raster,
                cfg.installations_x[0],
                cfg.installations_y[0],
                self.sprites["installation_1"],
            ),
            lambda r: r,  # return raster unchanged if false
            raster,
        )
        raster = jax.lax.cond(
            state.installations[1],
            lambda r: aj.render_at(
                raster,
                cfg.installations_x[1],
                cfg.installations_y[1],
                self.sprites["installation_2"],
            ),
            lambda r: r,
            raster,
        )
        raster = jax.lax.cond(
            state.installations[2],
            lambda r: aj.render_at(
                raster,
                cfg.installations_x[2],
                cfg.installations_y[2],
                self.sprites["installation_3"],
            ),
            lambda r: r,
            raster,
        )
        raster = jax.lax.cond(
            state.installations[3],
            lambda r: aj.render_at(
                raster,
                cfg.installations_x[3],
                cfg.installations_y[3],
                self.sprites["installation_4"],
            ),
            lambda r: r,
            raster,
        )
        raster = jax.lax.cond(
            state.installations[4],
            lambda r: aj.render_at(
                raster,
                cfg.installations_x[4],
                cfg.installations_y[4],
                self.sprites["installation_5"],
            ),
            lambda r: r,
            raster,
        )
        raster = jax.lax.cond(
            state.installations[5],
            lambda r: aj.render_at(
                raster,
                cfg.installations_x[5],
                cfg.installations_y[5],
                self.sprites["installation_6"],
            ),
            lambda r: r,
            raster,
        )

        # add solid white bullets
        bullet_sprite = _solid_sprite(
            cfg.bullet_width, cfg.bullet_height, (255, 255, 255)
        )

        def _draw_bullet(i, ras):
            alive = state.bullets_alive[i]
            bx, by = state.bullets[i, 0], state.bullets[i, 1]
            return jax.lax.cond(
                alive,
                lambda r: aj.render_at(r, bx, by, bullet_sprite),
                lambda r: r,
                ras,
            )

        raster = jax.lax.fori_loop(0, cfg.max_bullets, _draw_bullet, raster)

        # add enemies
        enemy_sprites = (
            self.sprites["long_enemy"],
            self.sprites["round_enemy"],
            self.sprites["small_enemy"],
        )

        def _draw_enemy(i, ras):
            state_i = state.enemies[i]
            active = state.enemies[i, 5] == 1
            ex = state.enemies[i, 0].astype(jnp.int32)
            ey = state.enemies[i, 1].astype(jnp.int32)
            flip = state.enemies[i, 2] < 0  # dx < 0 -> facing left
            enemy_type = state.enemies[i, 3]
            enemy_type = state_i[3].astype(jnp.int32)

            def _do(r):
                fns = (
                    lambda r: aj.render_at(
                        r, ex, ey, enemy_sprites[0], flip_horizontal=flip
                    ),
                    lambda r: aj.render_at(
                        r, ex, ey, enemy_sprites[1], flip_horizontal=flip
                    ),
                    lambda r: aj.render_at(
                        r, ex, ey, enemy_sprites[2], flip_horizontal=flip
                    ),
                )
                # chooses based on the enemy_type (int) the draw function (this change needed because cond doesn't work with different sprite sizes)
                return jax.lax.switch(enemy_type, fns, r)

            return jax.lax.cond(active, _do, lambda r: r, ras)

        raster = jax.lax.fori_loop(0, cfg.max_enemies, _draw_enemy, raster)

        # render the score
        max_digits = self.config.max_digits_for_score  # max amount of digits
        num_digits = jnp.where(
            state.score > 0,
            (
                jnp.ceil(
                    jnp.log10(state.score.astype(jnp.float32) + 1.0)
                ).astype(jnp.int32)
            ),
            1,
        )  # actual amount of digits

        score_digits = aj.int_to_digits(
            state.score, max_digits=max_digits
        )  # get digit array

        # Position (centered on top)
        digit_w = 8
        total_w = digit_w * num_digits
        score_x = (self.config.screen_width - total_w) // 2
        score_y = 5

        # Render score using the selective renderer
        raster = aj.render_label_selective(
            raster,
            score_x,
            score_y,
            score_digits,
            self.score_digit_sprites,
            max_digits - num_digits,  # skip 0s in front of score
            num_digits,  # show that many digits
            spacing=digit_w,
        )

        # 1) Pre-make two full‐height, static-shape beams:
        beam_light_blue = _solid_sprite(
            self.config.height_upper_beam, 3, (90, 204, 165)
        )
        beam_green = _solid_sprite(50, 3, (61, 151, 60))

        # 2) Helper to stack beams on top of each other
        def _draw_two_tone_beam(raster, x):
            r1 = aj.render_at(
                raster, x, self.config.start_beam, beam_light_blue
            )
            # then draw yellow from start_y downward (overwrites just the top segment)
            return aj.render_at(
                r1,
                x,
                self.config.start_beam + self.config.height_upper_beam,
                beam_green,
            )

        # 3) Draw plasma:
        def _handle_draw_plasma(i, raster):
            lane = state.enemies[i, 4].astype(jnp.int32)
            active = state.enemies[i, 5] == 1
            can_shoot = active & state.plasma_active[i]
            on_lane4 = lane == 3

            dx_i = state.enemies[i, 2]
            x_i = state.enemies[i, 0].astype(jnp.int32)
            half_w = (cfg.enemy_width[i] // 2).astype(jnp.int32)

            # If dx <0 enemy is moving from right to left. Plasma should be drawn at enemy_x plus half width
            # If dx > 0 enemy is moving from left to right. Plasma should be drawn at enemy_x plus enemy_x_width minus half width
            ex = jnp.where(
                dx_i < 0, x_i + half_w, x_i + cfg.enemy_width[i] - half_w
            )

            def _draw(r):
                return _draw_two_tone_beam(r, ex)

            return jax.lax.cond(
                on_lane4 & can_shoot, _draw, lambda r: r, raster
            )

        # 5) finally, run it across all enemies

        raster = jax.lax.fori_loop(
            0, cfg.max_enemies, _handle_draw_plasma, raster
        )

        return raster


class JaxAtlantis(
    JaxEnvironment[
        AtlantisState, AtlantisObservation, AtlantisInfo, AtlantisConstants
    ]
):
    """
    JAX-accelerated implementation of the classic Atari Atlantis game.

    This class provides a complete, high-performance implementation of Atlantis
    using JAX for GPU acceleration and JIT compilation. The game faithfully
    recreates the original mechanics while adding modern optimizations.

    Game Features:
    - Three-cannon defense system (left, center, right)
    - Four-lane enemy movement with queue-like behavior
    - Three enemy types with different speeds, sizes, and point values
    - Plasma beam attacks from enemies in the fourth lane
    - Installation revival system based on scoring (10000 per wave for one revival)
    - Progressive wave system with increasing difficulty
    - Authentic scoring system with bonus multipliers

    Technical Features:
    - Fully JAX-compatible with JIT compilation support
    - Vectorized collision detection and physics
    - Immutable state management for functional programming
    - Configurable parameters via GameConfig
    - Multiple reward function support for RL training
    - Frame skipping for training efficiency

    Scoring System (based on original Atari manual):
    - Large Gorgon Vessel: 100 points (center), 200 points (side cannons)
    - Gorgon Bandit Bomber: 1000 points (center), 2000 points (side cannons)
    - Wave completion bonus: 500 points per surviving installation
    - Installation revival: Every 10,000 points grants one revival credit

    Installation Revival Rules:
    - Command Post has highest revival priority
    - Credits carry over between waves if unused
    - Revival occurs at the end of each wave
    - Game ends when all installations destroyed and no credits remain

    Args:
        frameskip (int): Number of frames to skip between actions (default: 1)
        reward_funcs (list[callable]): Custom reward functions for RL training
        config (GameConfig): Game configuration parameters

    Attributes:
        config (GameConfig): Current game configuration
        frameskip (int): Frame skipping factor
        frame_stack_size (int): Number of frames to stack for observations
        reward_funcs (tuple): Tuple of reward functions for multi-objective RL
    """

    def __init__(
        self,
        frameskip: int = 1,
        reward_funcs: list[callable] = None,
        config: GameConfig | None = None,
    ):
        super().__init__()
        # Use provided config or create default configuration
        self.config = config or GameConfig()
        self.frameskip = frameskip
        self.frame_stack_size = 4  # Standard for Atari environments

        # Convert reward functions to tuple for JAX compatibility
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.renderer = Renderer_AtraJaxis(config=self.config)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
        ]

    def reset(
        self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)
    ) -> Tuple[AtlantisObservation, AtlantisState]:
        # --- empty tables ---
        empty_enemies = jnp.zeros((self.config.max_enemies, 6), dtype=jnp.int32)
        empty_bullets = jnp.zeros((self.config.max_bullets, 4), dtype=jnp.int32)
        empty_bullets_alive = jnp.zeros(
            (self.config.max_bullets,), dtype=jnp.bool_
        )
        empty_lanes = jnp.ones((4,), dtype=jnp.bool_)  # All lanes start free
        start_installations = jnp.ones(
            (6,), dtype=jnp.bool_
        )  # All installations start intact

        # Split PRNG key for spawn timer initialization and future use
        key, sub = jax.random.split(key)

        # Create initial game state with all systems reset
        new_state = AtlantisState(
            # Scoring system
            score=jnp.array(0, dtype=jnp.int32),
            reward=jnp.array(0, dtype=jnp.int32),
            score_spent=jnp.array(0, dtype=jnp.int32),
            wave=jnp.array(0, dtype=jnp.int32),  # Start with wave 0
            # Entity arrays
            enemies=empty_enemies,
            bullets=empty_bullets,
            bullets_alive=empty_bullets_alive,
            # Cannon control
            fire_cooldown=jnp.array(0, dtype=jnp.int32),
            fire_button_prev=jnp.array(False, dtype=jnp.bool_),
            # Enemy spawning system with random initial timer
            enemy_spawn_timer=jax.random.randint(
                sub,
                (),
                self.config.enemy_spawn_min_frames,
                self.config.enemy_spawn_max_frames + 1,
                dtype=jnp.int32,
            ),
            rng=key,
            lanes_free=empty_lanes,
            number_enemies_wave_remaining=jnp.array(
                self.config.wave_start_enemy_count, dtype=jnp.int32
            ),
            wave_end_cooldown_remaining=jnp.array(0, dtype=jnp.int32),
            # Defensive structures (all start intact)
            command_post_alive=jnp.array(True, dtype=jnp.bool_),
            installations=start_installations,
            # Plasma system (all enemies start with plasma capability)
            plasma_active=jnp.ones((self.config.max_enemies,), dtype=jnp.bool_),
        )

        # Generate initial observation
        obs = self._get_observation(new_state)
        return obs, new_state

    def _interpret_action(self, state, action) -> Tuple[bool, int]:
        """
        Translate action into control signals
        Returns two vars:

        fire_pressed: If any button is currently pressed
        cannon_idx: (0) left, (1) centre, (2) right or -1.
        """
        fire_pressed = (
            (action == Action.LEFTFIRE)
            | (action == Action.FIRE)
            | (action == Action.RIGHTFIRE)
        )
        # It is important to keep track if the button just got pressed
        # to prevent holding the button down and spamming bullets
        just_pressed = fire_pressed & (~state.fire_button_prev)
        can_shoot = (state.fire_cooldown == 0) & just_pressed

        middle_allowed = jnp.logical_or(
            action != Action.FIRE, state.command_post_alive
        )  # false if middle canon used, but already dead
        can_shoot = can_shoot & middle_allowed

        cannon_idx = jnp.where(
            can_shoot,
            jnp.where(
                action == Action.LEFTFIRE,
                0,
                jnp.where(
                    action == Action.FIRE,
                    1,
                    jnp.where(action == Action.RIGHTFIRE, 2, -1),
                ),
            ),
            -1,
        )
        return fire_pressed, cannon_idx

    def _spawn_bullet(self, state, cannon_idx):
        """Insert newly spawned bullet in first free slot"""
        cfg = self.config

        def _do_spawn(s):
            # To identify which slots are free
            # bullets_alive is a boolean array. If an entry is true, then it holds an active bullet
            # ~ inverts the boolean array, such that a slot is free, when bullets_alive[i] == False
            free_slots = ~s.bullets_alive
            slot_available = jnp.any(free_slots)  # at least one free?
            slot_idx = jnp.argmax(free_slots)  # first free slot

            # horizontal component dx:
            # - if cannon_idx == 0 (left), shoot rightwards -> +bullet_speed
            # - if cannond_idx == 2 (right), shoot leftwards -> -bullet_speed
            # else go straigt -> 0
            dx = jnp.where(
                cannon_idx == 0,  # true for left cannon
                cfg.bullet_speed,  # e.g. +3 pixels/frame
                jnp.where(
                    cannon_idx == 2,  # true for right cannon
                    -cfg.bullet_speed,  # e.g. -3 px
                    0,  # zero horizontal velocity
                ),
            )

            # vertical component dy:
            # - side bullets move slightly slower up than middle bullet Because origin is in top left, its negative
            dy = jnp.where(
                jnp.logical_or(cannon_idx == 0, cannon_idx == 2),
                -(cfg.bullet_speed - 1),
                -cfg.bullet_speed,
            )

            bullet_offset = jnp.array([7, 3, -1], dtype=jnp.int32)
            new_bullet = jnp.array(
                [
                    cfg.cannon_x[cannon_idx] + bullet_offset[cannon_idx],
                    cfg.cannon_y[cannon_idx],
                    dx,
                    dy,
                ],  # velocity
                dtype=jnp.int32,
            )

            # write into state
            def _write(s2):
                b2 = s2.bullets.at[slot_idx].set(new_bullet)
                a2 = s2.bullets_alive.at[slot_idx].set(True)
                return s2._replace(bullets=b2, bullets_alive=a2)

            # Conditionally write if a free slot exists
            return jax.lax.cond(slot_available, _write, lambda x: x, s)

        # Only attempt the spawn when a cannon actually fired this frame
        return jax.lax.cond(cannon_idx >= 0, _do_spawn, lambda x: x, state)

    def _update_cooldown(self, state, cannon_idx):
        """Reset after a shot or decrement the fire cooldown timer."""
        cfg = self.config
        new_cd = jnp.where(
            cannon_idx >= 0,  # -1 means no cannon fired
            jnp.array(cfg.fire_cooldown_frames, dtype=jnp.int32),
            jnp.maximum(state.fire_cooldown - 1, 0),
        )
        return state._replace(fire_cooldown=new_cd)

    def _move_bullets(self, state):
        """Move bullets by their velocity and deactivate offscreen bullets"""
        cfg = self.config

        # compute new x and y positions by adding the velocity dx and dy
        # state.bullets has shape (max_bullets, 4): (x,y,dx,dy)
        # [:, :2] takes all rows, but only columns 0 and 1 which are x and y
        # 2:4 then is dx and dy
        positions = state.bullets[:, :2] + state.bullets[:, 2:4]
        # Write updated position back into bullets array
        moved = state.bullets.at[:, :2].set(positions)

        # check if bullets are still onscreen
        in_bounds = (
            (positions[:, 0] >= 0)
            & (positions[:, 0] < cfg.screen_width)
            & (positions[:, 1] >= 0)
            & (positions[:, 1] < cfg.screen_height)
        )

        # a bullet only remains alive if it was already alive and still on-screen
        alive = state.bullets_alive & in_bounds
        return state._replace(bullets=moved, bullets_alive=alive)

    @staticmethod
    @jax.jit
    def _sample_speed(rng, wave):
        """
        Returns speed with a long left tail (mostly slow, rare fast).
        Uses a geometric distribution, shifting as wave increases.
        """
        # Higher waves -> lower p -> more fast enemies
        base_p = 0.8

        # adjust probability p for the geometric distribution
        # clip p to always stay between 0.3 and 0.95
        p = jnp.clip(base_p - 0.03 * wave, 0.3, 0.95)
        speed = jax.random.geometric(rng, p)
        max_speed = (
            wave + 1
        )  # limit speed. wave=0 -> max speed 1. wave=1 -> 2,...
        return jnp.minimum(speed, max_speed)

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_enemy(self, state: AtlantisState) -> AtlantisState:
        """
        • Decrement spawn-timer every frame if lane is free
        • When it reaches 0, try to insert one enemy into the first free
          slot of state.enemies
        • set to first lane
        • Pick direction with prng
        • After spawning (or if the screen is full) reset the timer to a
          new random value in min, max and advance the rng
        """

        cfg = self.config

        # helper that creates  a fresh timer value
        def _next_timer(rng):
            """Draw new integer in min, max inclusive"""
            return jax.random.randint(
                rng,
                (),
                cfg.enemy_spawn_min_frames,
                cfg.enemy_spawn_max_frames + 1,
                dtype=jnp.int32,
            )

        # check if the first lane is free
        lane_free = state.lanes_free[0]
        # Count down the timer if lane is free
        timer = jnp.where(
            lane_free, state.enemy_spawn_timer - 1, state.enemy_spawn_timer
        )

        # Split the current PRNG key into two new, independent keys
        #   rng_spawn will be used to draw random values for spawning enemies
        #   rng_after will be stored for the next frame’s randomness
        rng_spawn, rng_speed, rng_type, rng_after = jax.random.split(
            state.rng, 4
        )

        # if the timer is still bigger than 0, just update the timer and rng state
        def _no_spawn(s: AtlantisState) -> AtlantisState:
            return s._replace(enemy_spawn_timer=timer, rng=rng_after)

        def _spawn(s: AtlantisState) -> AtlantisState:
            # enemy has 5 entries, the last one (index 5) is the active_flag
            # if this value is 0, it means an enemy isn't active anymore
            # this can be because he either left the screen, or he was shot
            # the code returns a boolean array (active_flag == 0 -> true)
            free_slots = s.enemies[:, 5] == 0
            # check if at least one entry is true
            have_slot = jnp.any(free_slots)
            # get free slot index
            slot_idx = jnp.argmax(free_slots)

            # Choose a lane (rows in cfg.enemy_paths) and a direction.
            lane_idx = 0
            lane_y = cfg.enemy_paths[lane_idx]

            # Choose randomly the enemy type. Probabilities are 0.45 for type 0,0.45 for type 1,0.1  for type 2
            type_id = jax.random.choice(
                rng_type,
                a=jnp.array([0, 1, 2], dtype=jnp.int32),
                shape=(),
                p=jnp.array(
                    [
                        cfg.enemy_probabilities[0],
                        cfg.enemy_probabilities[1],
                        cfg.enemy_probabilities[2],
                    ],
                    dtype=jnp.float32,
                ),
            )

            # randomy decide the direction of the enemies, left or right
            go_left = jax.random.bernoulli(rng_spawn)  # True == left
            # if go_left is True, then set start x to the window_size + enemy_width
            # this ensures, that the enemy will spawn outside the visible area
            # if the value is false, spawn outside the visible area on the left side
            start_x = jnp.where(
                go_left,
                cfg.screen_width,
                -cfg.enemy_width[type_id],
            )

            # Set the direction
            speed = (
                self._sample_speed(rng_speed, s.wave)
                * cfg.enemy_speed_multipliers[type_id]
            )
            dx = jnp.where(go_left, -speed, speed)

            # debug.print(" Test DX {dx}", dx=dx)

            # dx = jnp.where(go_left, -cfg.enemy_speed, cfg.enemy_speed)

            # also sets the enemy to be active (last entry is 1)
            new_enemy = jnp.array(
                [start_x, lane_y, dx, type_id, 0, 1],
                dtype=jnp.int32,
            )

            def _write(write_s):
                updated_enemies = write_s.enemies.at[slot_idx].set(new_enemy)
                p2 = write_s.plasma_active.at[slot_idx].set(True)
                return write_s._replace(
                    enemies=updated_enemies, plasma_active=p2
                )

            # if enemies still has an empty slot, then write the new enemy
            # otherwise leave the state unchanged
            updated_state = jax.lax.cond(have_slot, _write, lambda x: x, s)

            # reset the timer
            new_timer = _next_timer(rng_after)
            # set first lane to full
            new_lanes = s.lanes_free.at[0].set(False)

            # decrease counter of spawnable enemies per wave
            updated_state = updated_state._replace(
                number_enemies_wave_remaining=(
                    s.number_enemies_wave_remaining - 1
                )
            )
            # jax.debug.print(f"Enemies remaining: {updated_state.number_enemies_wave_remaining}")

            return updated_state._replace(
                enemy_spawn_timer=new_timer, rng=rng_after, lanes_free=new_lanes
            )

        spawn_allowed = (timer <= 0) & (state.number_enemies_wave_remaining > 0)
        return jax.lax.cond(
            spawn_allowed,  # condition
            _spawn,  # If there are remaining enemies in the wave and time is 0, spawn
            _no_spawn,  # do not spawn
            state,
        )  # operands for the two functions

    # move all active enemies horizontally and deactive off-screen ones
    @partial(jax.jit, static_argnums=(0,))
    def _move_enemies(self, state: AtlantisState) -> AtlantisState:
        cfg = self.config
        enemies = state.enemies
        x_pos = enemies[:, 0]
        y_pos = enemies[:, 1]
        dx_vel = enemies[:, 2]
        enemy_ids = enemies[:, 3]
        lane_indices = enemies[:, 4]  # get current lanes of all enemies
        is_active = enemies[:, 5] == 1  # get active flags of all enemies
        number_lanes = cfg.enemy_paths.shape[0]

        # y always stays constant. just move x by adding dx
        new_pos = x_pos + dx_vel  # x + dx
        # -- enemies = x_pos.set(new_pos)  # write back

        # decide if an enemy is still on_screen
        # as long as a part of the enemy is still in the viewable area, the enemy stays alive
        # 1) check right edge > 0 -> enemies right edge hasnt completely passed the left edge of the screen
        # 2) check left edge < screen_width -> enemies left edge hasnt gone past the right edge of the screen
        on_screen = (new_pos + cfg.enemy_width[enemy_ids] > 0) & (
            new_pos < cfg.screen_width
        )

        # Identify enemies that are NOT on screen
        off_screen_enemies = ~on_screen
        inactive_enemies = ~is_active

        # identify all the enemies that move left (dx negative)
        # used for wrap-around/respawn position
        # if negative, spawn on right side. Otherwise on left side
        # but with an offset of enemy_width
        respawn_x = jnp.where(
            dx_vel < 0, cfg.screen_width, -cfg.enemy_width[enemy_ids]
        )

        # create array of length max_enemies
        # If the enemy is still active, it's lane-id gets incremented by 1
        # otherwise the lane gets set to 0
        # stores id of next lane for each enemy
        next_lanes = jnp.where(is_active, lane_indices + 1, 0)  # set dummy 0

        # Determine which enemies are allowed to advance into the next lane:
        # - Inactive enemies are always allowed to advance (the dont block)
        # - If the enemy would advance past the last lane (next_lanes >= number_lanes), block this.
        #   THis ensures, they are correctly deactivated after the last lane
        # - For active enemies withing bounds, only allow if the target lane is currently free
        # This logic ensures that only one enemy  can occupy a lane at a time.
        # If a faster enemy reaches the end of its lane but the next lane is occupied,
        # it will "wait" until the next lane becomes available
        # This creates a queue-like behaviour
        next_lane_free = (
            inactive_enemies  # already inactive
            | (
                next_lanes >= number_lanes
            )  # past the last lane -> Enemy will be deactivated later
            | (
                (next_lanes < number_lanes) & state.lanes_free[next_lanes]
            )  # only free lanes. Normal case
        )
        # Apply the new x positions only where off_screen_enemies is True
        # and for active enemies
        updated_x = jnp.where(
            (off_screen_enemies & next_lane_free & is_active),
            respawn_x,
            new_pos,  # Keep original x positions for on-screen enemies
        )

        # Then update the lanes
        updated_lanes = jnp.where(
            (off_screen_enemies & next_lane_free),
            lane_indices + 1,
            lane_indices,
        )

        # combine previous active flag with check for last lane
        # any enemy that's went through all four lanes gets deactivated
        flags = is_active & (updated_lanes < number_lanes)

        # Get the corresponding y-positions from enemy_paths
        lane_y_positions = jnp.where(
            updated_lanes < number_lanes,
            cfg.enemy_paths[updated_lanes],
            -cfg.enemy_height[enemy_ids],
        )

        updated_enemies = enemies.at[:, 0].set(updated_x)
        updated_enemies = updated_enemies.at[:, 1].set(lane_y_positions)
        updated_enemies = updated_enemies.at[:, 4].set(updated_lanes)
        updated_enemies = updated_enemies.at[:, 5].set(flags)

        # check if lanes are free now
        lane_masks = []
        # iterate through length of enemy_paths (4)
        # lane_mask checks if
        for i in range(len(cfg.enemy_paths)):
            # For each lane, check if any active enemy is in that lane
            lane_mask = (updated_enemies[:, 4] == i) & flags
            lane_is_occupied = jnp.any(lane_mask)
            lane_masks.append(~lane_is_occupied)  # True if lane is free

        free_lanes = jnp.array(lane_masks)
        return state._replace(enemies=updated_enemies, lanes_free=free_lanes)

    @partial(jax.jit, static_argnums=(0,))
    def _check_bullet_enemy_collision(
        self, state: AtlantisState
    ) -> AtlantisState:
        """
        Collision check between bullets and enemies

        Each bulllet/enemy an axis-aligned rectangle. Now:
        1. compute the four edges (lefet, right, top, bottom) for every bullet and every enemy
        2. Build two (BxE) boolean matrices for x-overlap and y-overlap
        3. compute AND of the two matrices and build the hit_matrix[b,e]. An entry is true, when bullet b and enemy e overlap in both X and Y
        4. ignore inactive bullets/enemies (through masking)
        5. reduce hit_mtarix to per-bullet and per_enemy "was hit?" flags
        6. deactive those objects
        7. update score
        """

        def handle_collision(l_hit_matrix, l_state: AtlantisState):
            # check if bullet collided with any enemy
            bullet_hit = jnp.any(l_hit_matrix, axis=1)  # (B,)
            # check if enemy was hit by any bullet
            enemy_hit = jnp.any(l_hit_matrix, axis=0)  # (E,)

            # which cannon was used (check horizontal velocity)
            dx = l_state.bullets[:, 2]
            side_cannon_hit = jnp.any(
                l_hit_matrix & (dx[:, None] != 0), axis=0
            )  # vector containing hits made with a side cannon

            # ---- score calculation ----
            types = l_state.enemies[:, 3]  # get all enemy types
            # calculate points per enemy (no matter if it was hit or not)
            # enemy type 0: center=100, side=200
            points_type_0 = jnp.where(
                types == 0,
                jnp.where(
                    side_cannon_hit,
                    2 * cfg.enemy_points[0],
                    cfg.enemy_points[0],
                ),
                0,
            )
            # enemy type 1: center=100, side=200
            points_type_1 = jnp.where(
                types == 1,
                jnp.where(
                    side_cannon_hit,
                    2 * cfg.enemy_points[1],
                    cfg.enemy_points[1],
                ),
                0,
            )
            # enemy type 2: center=1000, side=2000
            points_type_2 = jnp.where(
                types == 2,
                jnp.where(
                    side_cannon_hit,
                    2 * cfg.enemy_points[2],
                    cfg.enemy_points[2],
                ),
                0,
            )
            points_per_enemy = points_type_0 + points_type_1 + points_type_2

            # multiply points per enemy with enemy_hit to only get points of hit enemies; then sum points up
            points = jnp.sum(points_per_enemy * enemy_hit.astype(jnp.int32))
            # update score
            new_score = l_state.score + points

            # if at least 1 type 2 enemy is hit, remove all enemies
            type_2_killed = jnp.any(enemy_hit & (types == 2))
            enemy_hit = jnp.where(
                type_2_killed, jnp.ones_like(enemy_hit), enemy_hit
            )

            # deactivate bullets and enemies
            new_bullet_alive = l_state.bullets_alive & (~bullet_hit)

            new_enemy_flags = (l_state.enemies[:, 5] == 1) & (~enemy_hit)
            enemies_updated = l_state.enemies.at[:, 5].set(
                new_enemy_flags.astype(jnp.int32)
            )

            return l_state._replace(
                bullets_alive=new_bullet_alive,
                enemies=enemies_updated,
                score=new_score,
            )

        cfg = self.config

        bullets_x, bullets_y = state.bullets[:, 0], state.bullets[:, 1]  # (B,)
        enemies_x, enemies_y, type_ids = (
            state.enemies[:, 0],
            state.enemies[:, 1],
            state.enemies[:, 3].astype(jnp.int32),
        )  # (E,)

        # compute edge coordinates  for all rectangles
        # broadcasting with none inserts singleton axes so every
        # bullet is paired with every enemy
        b_left = bullets_x[:, None]
        b_right = (bullets_x + cfg.bullet_width)[:, None]
        b_top = bullets_y[:, None]
        b_bottom = (bullets_y + cfg.bullet_height)[:, None]

        # width per enemy
        enemy_widths = cfg.enemy_width[type_ids]
        enemy_heights = cfg.enemy_height[type_ids]

        # Enemy edges
        e_left = enemies_x[None, :]
        e_right = (enemies_x + enemy_widths)[None, :]
        e_top = enemies_y[None, :]
        e_bottom = (enemies_y + enemy_heights)[None, :]

        # True where bullets left < enemies right AND bullets right >  enemies left
        overlaps_x = (b_left < e_right) & (b_right > e_left)
        # ...
        overlaps_y = (b_top < e_bottom) & (b_bottom > e_top)

        # Matrix containing all enemies & bullets
        # True when both horizontal and vertical overlaps occur
        hit_matrix = overlaps_x & overlaps_y

        # Ignore inactive objects right away
        hit_matrix &= state.bullets_alive[:, None]
        hit_matrix &= (state.enemies[:, 5] == 1)[None, :]

        state = handle_collision(hit_matrix, state)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _update_wave(self, state: AtlantisState) -> AtlantisState:
        """
        Handle wave progression and installation revival system.

        This function manages:
        1. Wave completion detection
        2. Installation revival based on scoring (every 10,000 points)
        3. Next wave initialization with increased enemy count
        4. Bonus points for surviving installations (500 points each)

        Installation Revival Rules (from official Atari manual):
        - For every 10,000 points scored, one destroyed installation is restored
        - Command Post has rebuilding priority over other installations
        - Credits are saved across waves if no buildings were destroyed
        - Revival happens at the end of each wave

        Args:
            state: Current game state

        Returns:
            Updated game state with wave progression and potential revivals
        """
        cfg = self.config

        def _new_wave(s: AtlantisState) -> AtlantisState:
            """
            Start a new wave with installation revival and bonus scoring.
            """
            new_wave = s.wave + 1

            # Calculate bonus points for surviving installations (500 points each)
            surviving_installations = jnp.sum(s.installations.astype(jnp.int32))
            command_post_bonus = jnp.where(s.command_post_alive, 500, 0)
            wave_bonus = (surviving_installations * 500) + command_post_bonus

            # Calculate revival credits based on total score
            # Every 10,000 points grants one revival credit
            total_revival_credits = s.score // 10000
            used_credits = s.score_spent // 10000
            available_credits = total_revival_credits - used_credits

            # Determine how many installations need revival
            destroyed_installations = ~s.installations
            command_post_destroyed = ~s.command_post_alive
            total_destroyed = jnp.sum(
                destroyed_installations.astype(jnp.int32)
            ) + command_post_destroyed.astype(jnp.int32)

            # Calculate actual revivals (limited by credits and destroyed count)
            revivals_to_perform = jnp.minimum(
                available_credits, total_destroyed
            )

            # Revive installations with priority system:
            # 1. Command Post has highest priority
            # 2. Then installations in order (0, 1, 2, 3, 4, 5)

            # Revive command post first if destroyed and credits available
            revive_command_post = command_post_destroyed & (
                revivals_to_perform > 0
            )
            new_command_post_alive = s.command_post_alive | revive_command_post
            credits_after_command_post = jnp.where(
                revive_command_post,
                revivals_to_perform - 1,
                revivals_to_perform,
            )

            # Revive installations in order of priority
            new_installations = s.installations
            remaining_credits = credits_after_command_post

            def _revive_installation(i, carry):
                """Helper to revive installation i if credits available and installation destroyed."""
                installations, credits = carry

                # Check if this installation is destroyed and credits available
                can_revive = (~installations[i]) & (credits > 0)

                # Revive installation and decrement credits
                new_installations_i = installations.at[i].set(
                    installations[i] | can_revive
                )
                new_credits = jnp.where(can_revive, credits - 1, credits)

                return (new_installations_i, new_credits)

            # Apply revival to each installation in priority order
            final_installations, final_credits = jax.lax.fori_loop(
                0,
                6,
                _revive_installation,
                (new_installations, remaining_credits),
            )

            # Update score_spent to track used revival credits
            credits_used_this_wave = available_credits - final_credits
            new_score_spent = s.score_spent + (credits_used_this_wave * 10000)

            # Compute how many enemies next wave should have
            next_count = cfg.wave_start_enemy_count + new_wave * 2

            return s._replace(
                wave=new_wave,
                wave_end_cooldown_remaining=jnp.array(
                    cfg.wave_end_cooldown, jnp.int32
                ),
                number_enemies_wave_remaining=next_count,
                score=s.score + wave_bonus,  # Add survival bonus
                score_spent=new_score_spent,  # Track spent revival credits
                command_post_alive=new_command_post_alive,  # Apply command post revival
                installations=final_installations,  # Apply installation revivals
            )

        def _same_wave(s: AtlantisState) -> AtlantisState:
            """Continue current wave without changes."""
            return s

        # Start new wave if no enemies remaining and screen is empty
        wave_complete = (state.number_enemies_wave_remaining == 0) & (
            ~jnp.any(state.enemies[:, 5] == 1)
        )

        return jax.lax.cond(
            wave_complete,
            _new_wave,
            _same_wave,
            state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _cooldown_finished(self, state: AtlantisState) -> Array:
        return state.wave_end_cooldown_remaining == 0

    @partial(jax.jit, static_argnums=0)
    def _handle_plasma_hit(self, state: AtlantisState) -> AtlantisState:
        """
        Handle an incoming plasma beam from lane-4 enemies.

        This checks whether any active lane-4 enemy fires a beam that overlaps
        the central command post or one of the six installations, and if so,
        knocks it out and disables that enemy’s plasma until it leaves the screen.

        Steps:
          1. Identify which lane-4 enemies are active and still allowed to fire.
          2. Compute each shooter’s current beam X (at the enemy’s sprite center).
          3. Reconstruct the beam’s previous X by subtracting its dx (to cover high-speed passes).
          4. Form the inclusive interval [lo, hi] between old and new beam positions.
          5. Build the list of target X-coordinates: central command post + installations.
          6. Check which installment (if any) lies within [lo, hi].
          7. If the center post is still alive and is hit first, knock it out.
          8. Otherwise, if the center post is already down, knock out the installation hit.
          9. Disable this shooter’s plasma until it goes off-screen.

        """
        cfg = self.config

        # Find out enemy in 4th lane which currently is using plasma
        is_lane4 = (state.enemies[:, 4] == 3) & (state.enemies[:, 5] == 1)
        can_fire = is_lane4 & state.plasma_active
        shooter_idx = jnp.argmax(can_fire)
        shooter_fired = jnp.any(can_fire)

        type_ids = state.enemies[:, 3].astype(jnp.int32)

        dx_i = state.enemies[:, 2]
        x_i = state.enemies[:, 0].astype(jnp.int32)
        half_w = cfg.enemy_width[type_ids] // 2

        # If dx <0 enemy is moving from right to left. Plasma should be drawn at enemy_x plus half width
        # If dx > 0 enemy is moving from left to right. Plasma should be drawn at enemy_x plus enemy_x_width minus half width
        centers = jnp.where(
            dx_i < 0, x_i + half_w, x_i + cfg.enemy_width[type_ids] - half_w
        )

        # We try to simulate the beams last position. It is important because at higher speeds, we cannot have a exact
        # alignment of central canon and installment for knocking them out
        old_centers = centers - state.enemies[:, 2]
        beam_old = jnp.where(can_fire, old_centers, -1)
        beam_new = jnp.where(can_fire, centers, -1)

        # Now find the maximum (i.e. the only) beam among shooters
        beam_old = jnp.max(beam_old)  # -1 if none
        beam_new = jnp.max(beam_new)

        # Build 7 possible targets, 6 installments and one central canon
        # If enemy comes from left to right, collision occurs when plasma hits the right edge of the installation
        # If enemy comes from right to left, collision occurs when plasma hits the left edge of the installation
        # Although, in the real game, the enemy is shot down on first contact, since we don't have the animations right
        # now, we implemented this extra so that it looks good and real while playing

        dx_shooter = state.enemies[shooter_idx, 2]

        # For installations: hit the approaching edge based on enemy direction
        inst_targets = jnp.where(
            dx_shooter > 0,  # enemy moving left to right
            cfg.installations_x + cfg.installations_width,  # hit right edge
            cfg.installations_x,  # hit left edge
        )

        # Same logic for central cannon
        cmd_target = jnp.where(
            dx_shooter > 0,  # enemy moving left to right
            cfg.cannon_x[1] + cfg.cannon_width,  # hit right edge
            cfg.cannon_x[1],  # hit left edge
        )

        targets = jnp.concatenate([jnp.array([cmd_target]), inst_targets], 0)

        # Directional collision detection: check if beam has reached each target
        # For left-to-right movement: beam_new >= target AND beam_old < target
        # For right-to-left movement: beam_new <= target AND beam_old > target
        left_to_right_hit = (beam_new >= targets) & (beam_old < targets)
        right_to_left_hit = (beam_new <= targets) & (beam_old > targets)

        hit_mask = jnp.where(
            dx_shooter > 0,  # moving left to right
            left_to_right_hit,
            right_to_left_hit,
        )

        any_hit = jnp.any(hit_mask)
        hit_index = jnp.argmax(hit_mask)  # first match

        # 5) Decide who to kill
        kill_cmd = (
            shooter_fired
            & any_hit
            & (hit_index == 0)
            & state.command_post_alive
        )

        inst_idx = hit_index - 1  # maps 1→install[0], …, 6→install[5]
        inst_alive = (
            (inst_idx >= 0)
            & (inst_idx < inst_targets.shape[0])
            & state.installations[inst_idx]
        )
        kill_inst = (
            shooter_fired & any_hit & (~state.command_post_alive) & inst_alive
        )

        def _handle_hit(s: AtlantisState) -> AtlantisState:
            # a) knock out command post
            s1 = jax.lax.cond(
                kill_cmd,
                lambda st: st._replace(command_post_alive=False),
                lambda st: st,
                s,
            )

            # b) knock out the installation
            def _kill_one(st: AtlantisState) -> AtlantisState:
                return st._replace(
                    installations=st.installations.at[inst_idx].set(False)
                )

            s2 = jax.lax.cond(
                kill_inst,
                _kill_one,
                lambda st: st,
                s1,
            )

            # c) disable this shooter’s plasma if either the command post OR an installation was killed
            disable_plasma = kill_inst | kill_cmd
            return jax.lax.cond(
                disable_plasma,
                lambda st: st._replace(
                    plasma_active=st.plasma_active.at[shooter_idx].set(False)
                ),
                lambda st: st,
                s2,
            )

        return jax.lax.cond(
            shooter_fired & any_hit,
            _handle_hit,
            lambda st: st,
            state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _refresh_plasma_active(self, state: AtlantisState) -> AtlantisState:
        # compute each enemy’s x‐position
        new_pos = state.enemies[:, 0]
        # on_screen == True if any part of the enemy is still inside [0, screen_width)
        on_screen = (
            new_pos
            + self.config.enemy_width[state.enemies[:, 3].astype(jnp.int32)]
            > 0
        ) & (new_pos < self.config.screen_width)
        # whenever *off* screen, re‐enable that slot’s plasma_active bit
        allowed = jnp.where(on_screen, state.plasma_active, True)
        return state._replace(plasma_active=allowed)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: AtlantisState, action: chex.Array
    ) -> Tuple[AtlantisObservation, AtlantisState, float, bool, AtlantisInfo]:
        previous_state = state

        def _pause_step(s: AtlantisState) -> AtlantisState:
            # reduce pause cooldown
            s = s._replace(
                wave_end_cooldown_remaining=jnp.maximum(
                    s.wave_end_cooldown_remaining - 1, 0
                )
            )
            s = self._move_bullets(s)
            return self._update_wave(s)

        def _wave_step(s: AtlantisState) -> AtlantisState:
            # input handling
            fire_pressed, cannon_idx = self._interpret_action(s, action)

            # bullets
            s = self._spawn_bullet(s, cannon_idx)
            s = self._update_cooldown(s, cannon_idx)._replace(
                fire_button_prev=fire_pressed
            )

            # enemies
            s = self._spawn_enemy(s)

            # motion & collisions
            s = self._move_bullets(s)
            s = self._move_enemies(s)
            s = self._check_bullet_enemy_collision(s)
            s = self._refresh_plasma_active(s)
            s = self._handle_plasma_hit(s)

            # check if wave quota exhausted → start pause
            s = self._update_wave(s)
            return s

        state = jax.lax.cond(
            self._cooldown_finished(state), _wave_step, _pause_step, state
        )
        observation = self._get_observation(state)
        done = self._get_done(state)
        all_rewards = self._get_all_reward(previous_state, state)
        info = self._get_info(state, all_rewards)
        new_reward = state.score - previous_state.score
        state = state._replace(reward=new_reward)

        return observation, state, state.reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AtlantisState) -> AtlantisObservation:
        cfg = self.config

        # get types of enemies
        type_ids = state.enemies[:, 3].astype(jnp.int32)

        # Get the positions and dimensions of the enemies
        # set inactive enemies to -1 or 0
        enemy_alive = state.enemies[:, 5] == 1
        enemy_x = jnp.where(
            enemy_alive, state.enemies[:, 0].astype(jnp.int32), -1
        )
        enemy_y = jnp.where(
            enemy_alive, state.enemies[:, 1].astype(jnp.int32), -1
        )
        enemy_w = jnp.where(
            enemy_alive, cfg.enemy_width[type_ids].astype(jnp.int32), 0
        )
        enemy_h = jnp.where(
            enemy_alive, cfg.enemy_height[type_ids].astype(jnp.int32), 0
        )
        enemy_pos = EntityPosition(
            enemy_x, enemy_y, enemy_w, enemy_h, enemy_alive
        )

        # Get the positions and dimensions of the bullets
        # set inactive bullets to -1 or 0
        bullet_alive = state.bullets_alive
        bullet_x = jnp.where(
            bullet_alive, state.bullets[:, 0].astype(jnp.int32), -1
        )
        bullet_y = jnp.where(
            bullet_alive, state.bullets[:, 1].astype(jnp.int32), -1
        )
        bullet_w = jnp.where(
            bullet_alive,
            jnp.full((cfg.max_bullets,), cfg.bullet_width, dtype=jnp.int32),
            0,
        )
        bullet_h = jnp.where(
            bullet_alive,
            jnp.full((cfg.max_bullets,), cfg.bullet_height, dtype=jnp.int32),
            0,
        )
        bullet_pos = EntityPosition(
            bullet_x, bullet_y, bullet_w, bullet_h, bullet_alive
        )

        return AtlantisObservation(
            score=state.score,
            enemy=enemy_pos,
            bullet=bullet_pos,
            installations_alive=state.installations,
            command_post_alive=state.command_post_alive,
        )

    def observation_space(self) -> spaces.Dict:
        cfg = self.config

        def entity_space(n: int, w_max: int, h_max: int) -> spaces.Dict:
            return spaces.Dict(
                {
                    "x": spaces.Box(
                        low=-w_max,
                        high=cfg.screen_width,
                        shape=(n,),
                        dtype=jnp.int32,
                    ),
                    "y": spaces.Box(
                        low=-h_max,
                        high=cfg.screen_height,
                        shape=(n,),
                        dtype=jnp.int32,
                    ),
                    "width": spaces.Box(
                        low=0, high=w_max, shape=(n,), dtype=jnp.int32
                    ),
                    "height": spaces.Box(
                        low=0, high=h_max, shape=(n,), dtype=jnp.int32
                    ),
                    "alive": spaces.Box(
                        low=0, high=1, shape=(n,), dtype=jnp.int32
                    ),
                }
            )

        return spaces.Dict(
            {
                "score": spaces.Box(
                    low=0,
                    high=(10**cfg.max_digits_for_score) - 1,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "enemy": entity_space(
                    n=cfg.max_enemies,
                    w_max=int(jnp.max(cfg.enemy_width).item()),
                    h_max=int(jnp.max(cfg.enemy_height).item()),
                ),
                "bullet": entity_space(
                    n=cfg.max_bullets,
                    w_max=int(cfg.bullet_width),
                    h_max=int(cfg.bullet_height),
                ),
                "installations_alive": spaces.Box(
                    low=0,
                    high=1,
                    shape=(6,),
                    dtype=jnp.int32,
                ),
                "command_post_alive": spaces.Box(
                    low=0,
                    high=1,
                    shape=(),
                    dtype=jnp.int32,
                ),
            }
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: EnvObs) -> jnp.ndarray:
        def _flat(ep: EntityPosition) -> jnp.ndarray:
            return jnp.concatenate(
                [
                    jnp.ravel(ep.x).astype(jnp.int32),
                    jnp.ravel(ep.y).astype(jnp.int32),
                    jnp.ravel(ep.width).astype(jnp.int32),
                    jnp.ravel(ep.height).astype(jnp.int32),
                    jnp.ravel(ep.alive).astype(jnp.int32),  # booleans -> 0,1
                ],
                axis=0,
            )

        return jnp.concatenate(
            [
                jnp.atleast_1d(obs.score).astype(jnp.int32),
                _flat(obs.enemy),
                _flat(obs.bullet),
                obs.installations_alive.astype(jnp.int32),
                jnp.atleast_1d(obs.command_post_alive.astype(jnp.int32)),
            ],
            axis=0,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(
        self, state: AtlantisState, all_rewards: chex.Array = None
    ) -> AtlantisInfo:
        enemies_alive = jnp.sum((state.enemies[:, 5] == 1).astype(jnp.int32))
        bullets_alive = jnp.sum(state.bullets_alive.astype(jnp.int32))

        return AtlantisInfo(
            score=state.score,
            wave=state.wave,
            enemies_alive=enemies_alive,
            bullets_alive=bullets_alive,
            enemies_remaining_in_wave=state.number_enemies_wave_remaining,
            wave_cooldown_remaining=state.wave_end_cooldown_remaining,
            command_post_alive=state.command_post_alive,
            installations_alive=state.installations,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(
        self, previous_state: AtlantisState, state: AtlantisState
    ):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [
                reward_func(previous_state, state)
                for reward_func in self.reward_funcs
            ]
        )
        return rewards

    def _get_reward(
        self, previous_state: AtlantisState, state: AtlantisState
    ):
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AtlantisState) -> jnp.bool_:
        """
        Game is done when:
          1) Score has reached the maximum representable (i.e. max_digits_for_score), OR
          2) The central command post is destroyed and all installations are destroyed.
        """
        # 1) Max‐score condition
        max_score = 10**self.config.max_digits_for_score
        reached_max = state.score >= max_score  # bool scalar

        # 2) All defenses down?
        cmd_alive = state.command_post_alive  # bool scalar
        any_install_alive = jnp.any(
            state.installations
        )  # True if at least one installation remains
        defenses_down = (~cmd_alive) & (~any_install_alive)

        # 3) Final done flag
        done = reached_max | defenses_down

        # jax.debug.print("[_get_done] score={}|{}  cmd_alive={}  any_inst_alive={}  → done={}", state.score, max_score, cmd_alive, any_install_alive, done)

        return done

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for Atlantis.
        Actions are:
        0: NOOP
        1: FIRE
        2: RIGHTFIRE
        3: LEFTFIRE
        """
        return spaces.Discrete(len(self.action_set))

    def render(self, state: AtlantisState) -> jnp.ndarray:
        return self.renderer.render(state)

    def image_space(self) -> spaces.Box:
        cfg = self.config
        return spaces.Box(
            low=0,
            high=255,
            shape=(cfg.screen_height, cfg.screen_width, 3),
            dtype=jnp.uint8,
        )


# Keyboard inputs
def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    # up = keys[pygame.K_w] or keys[pygame.K_UP]
    # down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]

    if right and fire:
        return jnp.array(Action.RIGHTFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if fire:
        return jnp.array(Action.FIRE)

    return jnp.array(Action.NOOP)


def main():
    config = GameConfig()
    pygame.init()
    screen = pygame.display.set_mode(
        (
            config.screen_width * config.scaling_factor,
            config.screen_height * config.scaling_factor,
        )
    )
    pygame.display.set_caption("Atlantis")
    clock = pygame.time.Clock()

    game = JaxAtlantis(config=config)

    renderer = Renderer_AtraJaxis(config=config)
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    # (curr_state, _) = jitted_reset()
    (_, curr_state) = jitted_reset()

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        (obs, curr_state, _, _, _) = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                (obs, curr_state, _, _, _) = jitted_step(curr_state, action)
                # print("Enemies remaining:", int(curr_state.number_enemies_wave_remaining))
                # print("Current wave: ", int(curr_state.wave))
                # print(f"Timout: {int(curr_state.wave_end_cooldown_remaining)}")
                # dx_list = curr_state.enemies[:, 2][curr_state.enemies[:, 5] == 1].tolist()
                # print("Active enemy dx’s:", dx_list)

        # Render and display
        raster = renderer.render(curr_state)

        # aj.update_pygame(
        #    screen,
        #    raster,
        #    config.scaling_factor,
        #    config.screen_width,
        #    config.screen_height,
        # )

        counter += 1
        # FPS
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()


# --------------------- Script to visualize sprites ---------------------

# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# def main2():
#     # Pfad zur .npy Datei relativ zu diesem Skript
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     image_path = os.path.join(
#         script_dir,
#         "sprites", "atlantis", "0.npy"
#     )
#
#     # .npy laden (Form (H, W, 4) mit RGBA)
#     img = np.load(image_path)
#
#     # Figure und Achse anlegen, beide mit transparentem Hintergrund
#     fig, ax = plt.subplots(figsize=(6, 6),
#                            facecolor="none")        # Figure durchsichtig
#     ax.set_facecolor("none")                         # Achse durchsichtig
#
#     # Bild anzeigen (RGBA wird unterstützt)
#     ax.imshow(img)
#     ax.axis("off")  # keine Achsenlinien
#
#     # Interaktive Anzeige
#     plt.show()
#
#
# if __name__ == "__main__":
#     main2()
