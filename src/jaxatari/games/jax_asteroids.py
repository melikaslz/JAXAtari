"""
authors: Paula Troszt, Ernst Christian Böhringer, Aiman Sammy Rahlf
"""

import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional
import jax.lax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class AsteroidsConstants(NamedTuple):
    # Constants for game environment
    WIDTH: int = 160
    HEIGHT: int = 210

    # Player position can be "in between" pixels.
    # This is not visible on screen but relevant for calculation.
    # Internal representations of the player position must first be converted to screen coordinates.
    # A change of a coordinate value of this internal position by any value below 256
    # might not be visible in game if no pixel-boundary is crossed with said change.

    # Object sizes (width, height)
    PLAYER_SIZE: Tuple[int, int] = (5, 10)
    MISSILE_SIZE: Tuple[int, int] = (1, 2)
    ASTEROID_SIZE_L: Tuple[int, int] = (16, 28)
    ASTEROID_SIZE_M: Tuple[int, int] = (8, 15)
    ASTEROID_SIZE_S: Tuple[int, int] = (4, 8)

    # Asteroid color and size constants
    BROWN: int = 0 # 180, 122, 48
    GREY: int = 1 # 214, 214, 214 
    LIGHT_BLUE: int = 2 # 117, 181, 239
    LIGHT_YELLOW: int = 3 # 187, 187, 53
    PINK: int = 4 # 184, 70, 162
    PURPLE: int = 5 # 104, 72, 198
    RED: int = 6 # 184, 50, 50
    YELLOW: int = 7 # 136, 146, 62

    INACTIVE: int = 0
    LARGE_1: int = 1
    LARGE_2: int = 2
    MEDIUM: int = 3
    SMALL: int = 4

    # Start positions
    INITIAL_PLAYER_X: int = int(256/4 * (WIDTH - PLAYER_SIZE[0]))
    INITIAL_PLAYER_Y: int = int(256/4 * (HEIGHT - PLAYER_SIZE[1]))
    INITIAL_PLAYER_ROTATION: int = 0
    INITIAL_ASTEROID_STATES: chex.Array = jnp.array([
        [36, 24, 1, LARGE_2, BROWN],
        [124, 153, 3, LARGE_2, GREY],
        [48, 148, 1, LARGE_1, GREY],
        [126, 26, 3, LARGE_1, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY],
        [0, 0, 0, INACTIVE, GREY]
    ])

    # Rendering constants
    WALL_COLOR: Tuple[int, int, int] = (0, 0, 0)
    WALL_TOP_HEIGHT: int = 18
    WALL_BOTTOM_HEIGHT: int = 15

    # Game constants
    STARTING_LIVES: int = 4
    POINTS_PER_LIFE: int = 5000
    MAX_LIVES: int = 9
    MAX_SCORE: int = 100000
    SAFE_ZONE = (20, 34)
    MAX_ENTITY_X: int = WIDTH - 1
    MAX_ENTITY_Y: int = HEIGHT - WALL_BOTTOM_HEIGHT
    MIN_ENTITY_X: int = 0
    MIN_ENTITY_Y: int = WALL_TOP_HEIGHT

    # Player constants
    MAX_PLAYER_SPEED: int = 64 * 256 - 1
    MAX_PLAYER_X: int = int((WIDTH * 256 - 1)/2)
    MAX_PLAYER_Y: int = int(((HEIGHT - WALL_BOTTOM_HEIGHT) * 256 - 1)/2)
    MIN_PLAYER_X: int = 0
    MIN_PLAYER_Y: int = int((WALL_TOP_HEIGHT * 256 - 1)/2)
    RESPAWN_DELAY: int = 136
    H_SPACE_DELAY: int = 62
    ACCEL_PER_ROTATION: chex.Array = jnp.array([
        (0, -127),
        (-49, -117),
        (-90, -90),
        (-117, -49),
        (-127, 0),
        (-117, 49),
        (-90, 90),
        (-49, 117),
        (0, 127),
        (49, 117),
        (90, 90),
        (117, 49),
        (127, 0),
        (117, -49),
        (90, -90),
        (49, -117)
    ])

    # Missile constants
    MISSILE_LIFESPAN: int = 24
    MISSILE_OFFSET_PER_ROTATION: chex.Array = jnp.array([
        (0, 1),
        (-1, 0),
        (-3, 0),
        (-3, 2),
        (-4, 3),
        (-3, 4),
        (-3, 6),
        (-1, 6),
        (0, 7),
        (2, 6),
        (4, 6),
        (4, 4),
        (5, 3),
        (4, 2),
        (4, 0),
        (2, 0)
        ])
    MISSILE_SPEED_PER_ROTATION: chex.Array = jnp.array([
        (0, -4),
        (-1, -3),
        (-3, -3),
        (-3, -1),
        (-4, 0),
        (-3, 1),
        (-3, 3),
        (-1, 3),
        (0, 4),
        (1, 3),
        (3, 3),
        (3, 1),
        (4, 0),
        (3, -1),
        (3, -3),
        (1, -3)
    ])

    # Asteroid constants
    ASTEROID_SPEED: Tuple[int, int] = (2, 1)
    ASTEROID_BORDER_LEFT: int = int(MIN_ENTITY_X + (MAX_ENTITY_X-MIN_ENTITY_X)/3)
    ASTEROID_BORDER_RIGHT: int = int(MAX_ENTITY_X - (MAX_ENTITY_X-MIN_ENTITY_X)/3)
    MAX_NUMBER_OF_ASTEROIDS: int = 17
    NEW_ASTEROIDS_COUNT: int = 6

# immutable state container
class AsteroidsState(NamedTuple):

    player_x: chex.Array
    player_y: chex.Array
    player_speed_x: chex.Array
    player_speed_y: chex.Array
    player_rotation: chex.Array

    missile_states: chex.Array # (2, 6) array with (x, y, speed_x, speed_y, rotation, lifespan) for each missile
    asteroid_states: chex.Array # (17, 5) array with (x, y, rotation, size, color) for each asteroid

    missile_rdy: chex.Array # tracks whether the player can fire a missile
    colliding_asteroids: chex.Array # (3, 3) array with (index, size, animation_color) for each asteroid

    score: chex.Array
    lives: chex.Array
    respawn_timer: chex.Array # used to delay player reset after losing a life

    side_step_counter: chex.Array # tracks when asteroids perform movement on the x-axis
    step_counter: chex.Array
    rng_key: chex.PRNGKey

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    rotation: jnp.ndarray
    active: jnp.ndarray

class AsteroidsObservation(NamedTuple):
    player: EntityPosition # (x, y, width, height, rotation, active)
    missiles: jnp.ndarray  # shape (2, 6) - 2 missiles, each with (x, y, width, height, rotation, active)
    asteroids: jnp.ndarray # shape (17, 6) - 17 asteroids, each with (x, y, width, height, rotation, active)

    score: jnp.ndarray
    lives: jnp.ndarray

class AsteroidsInfo(NamedTuple):
    score: chex.Array
    step_counter: chex.Array
    all_rewards: chex.Array

class JaxAsteroids(JaxEnvironment[AsteroidsState, AsteroidsObservation, AsteroidsInfo, AsteroidsConstants]):
    def __init__(self, consts: AsteroidsConstants = None, reward_funcs: list[callable]=None):
        consts = consts or AsteroidsConstants()
        super().__init__(consts)
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = jnp.array([
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
            Action.UPLEFT,
            Action.UPRIGHT,
            Action.UPFIRE,
            Action.DOWNFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.UPLEFTFIRE,
            Action.UPRIGHTFIRE
        ])
        self.obs_size = 1*6 + 2*6 + self.consts.MAX_NUMBER_OF_ASTEROIDS*6 + 2
        self.renderer = AsteroidsRenderer(consts)

    @partial(jax.jit, static_argnums=(0,))
    def decel_func(self, speed):
        """
        Calculates the resistance applied to the player when not using thrusters
        based on the current speed
        """
        return -(2*(jnp.sign(speed)*(jnp.abs(speed)//256)) + jnp.sign(speed))

    @partial(jax.jit, static_argnums=(0,))
    def speed_func(self, speed):
        """
        Converts speed values in memory to speed in sub-pixels per step
        """
        return jnp.sign(speed)*(jnp.abs(speed)//32)

    @partial(jax.jit, static_argnums=(0,))
    def final_pos(self, min_pos, max_pos, pos):
        """
        Handles wrap-around
        """
        return ((pos - min_pos)%(max_pos - min_pos)) + min_pos

    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self,
        state_player_x,
        state_player_y,
        state_player_speed_x,
        state_player_speed_y,
        state_player_rotation,
        action,
        state_respawn_timer,
        rng_key
    ):
        # get pressed buttons
        left = jnp.logical_or(
            jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE),
            jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE))
        right = jnp.logical_or(
            jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE),
            jnp.logical_or(action == Action.UPRIGHT, action == Action.UPRIGHTFIRE))
        up = jnp.logical_or(
            jnp.logical_or(
                jnp.logical_or(action == Action.UP, action == Action.UPRIGHT),
                jnp.logical_or(action == Action.UPLEFT, action == Action.UPFIRE)),
            jnp.logical_or(action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE))
        down = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)

        player_x = state_player_x
        player_y = state_player_y
        player_speed_x = state_player_speed_x
        player_speed_y = state_player_speed_y
        player_rotation = state_player_rotation

        # rotate player left
        player_rotation = jax.lax.cond(
            left,
            lambda: (player_rotation+1)%16,
            lambda: player_rotation
        )

        # rotate player right
        player_rotation = jax.lax.cond(
            right,
            lambda: (player_rotation-1)%16,
            lambda: player_rotation
        )

        # get resistance for movement without thrusters
        decel_x = self.decel_func(player_speed_x)
        decel_y = self.decel_func(player_speed_y)

        # get acceleration for movement with thrusters
        accel_x = self.consts.ACCEL_PER_ROTATION[state_player_rotation][0]
        accel_y = self.consts.ACCEL_PER_ROTATION[state_player_rotation][1]

        # determine whether thrusters affect speed on the corresponding axis
        adj_speed_x = jnp.logical_and(
            jnp.logical_and(up, jnp.abs(player_speed_x + accel_x) < self.consts.MAX_PLAYER_SPEED),
            jnp.logical_not(player_rotation%8 == 0))
        adj_speed_y = jnp.logical_and(
            jnp.logical_and(up, jnp.abs(player_speed_y + accel_y) < self.consts.MAX_PLAYER_SPEED),
            jnp.logical_not((player_rotation-4)%8 == 0))

        # calculate new player speed
        player_speed_x = jax.lax.cond(
            adj_speed_x,
            lambda: player_speed_x + accel_x,
            lambda: player_speed_x
        )
        player_speed_x = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_x), jnp.abs(player_speed_x) > jnp.abs(decel_x)),
            lambda: player_speed_x + decel_x,
            lambda: player_speed_x
        )
        player_speed_x = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_x), jnp.abs(player_speed_x) <= jnp.abs(decel_x)),
            lambda: 0,
            lambda: player_speed_x
        )

        player_speed_y = jax.lax.cond(
            adj_speed_y,
            lambda: player_speed_y + accel_y,
            lambda: player_speed_y
        )
        player_speed_y = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_y), jnp.abs(player_speed_y) > jnp.abs(decel_y)),
            lambda: player_speed_y + decel_y,
            lambda: player_speed_y
        )
        player_speed_y = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_y), jnp.abs(player_speed_y) <= jnp.abs(decel_y)),
            lambda: 0,
            lambda: player_speed_y
        )

        # calculate the change in position depending on the new speed
        displace_x = self.speed_func(player_speed_x)
        displace_y = self.speed_func(player_speed_y)

        # calculate new player position
        player_x = self.final_pos(self.consts.MIN_PLAYER_X, self.consts.MAX_PLAYER_X, player_x + displace_x)
        player_y = self.final_pos(self.consts.MIN_PLAYER_Y, self.consts.MAX_PLAYER_Y, player_y + displace_y)

        # hyperspace
        key, subkey_x = jax.random.split(rng_key)
        key, subkey_y = jax.random.split(rng_key)
        new_player_x = jax.random.randint(subkey_x, [], self.consts.MIN_PLAYER_X, self.consts.MAX_PLAYER_X+1)
        new_player_y = jax.random.randint(subkey_y, [], self.consts.MIN_PLAYER_Y, self.consts.MAX_PLAYER_Y+1)

        respawn_timer, player_x, player_y, player_speed_x, player_speed_y = jax.lax.cond(
            down,
            lambda: (self.consts.RESPAWN_DELAY+1, new_player_x, new_player_y, 0, 0),
            lambda: (state_respawn_timer, player_x, player_y, player_speed_x, player_speed_y),
        )

        return jax.lax.cond(
            state_respawn_timer <= 0,
            lambda: (player_x, player_y, player_speed_x, player_speed_y,
                     player_rotation, respawn_timer, key),
            lambda: (state_player_x, state_player_y, state_player_speed_x, state_player_speed_y,
                     state_player_rotation, state_respawn_timer, rng_key)
        )

    @partial(jax.jit, static_argnums=(0,))
    def to_screen_pos(self, player_pos):
        """
        Converts a subpixel position to a screen position
        """
        return jnp.sign(player_pos)*(jnp.abs(player_pos)//256)*2

    @partial(jax.jit, static_argnums=(0,))
    def missile_step(
        self,
        asteroids_state,
        cur_player_x,
        cur_player_y,
        player_speed_x,
        player_speed_y,
        cur_player_rotation,
        action,
        player_active
    ):
        # get pressed buttons
        space = jnp.logical_or(jnp.logical_or(
                            jnp.logical_or(action == Action.FIRE, action == Action.UPFIRE),
                            jnp.logical_or(action == Action.LEFTFIRE, action == Action.RIGHTFIRE)),
                            jnp.logical_or(jnp.logical_or(action == Action.DOWNFIRE, action == Action.UPLEFTFIRE),
                            action == Action.UPRIGHTFIRE))

        missile_rdy = asteroids_state.missile_rdy
        missile_1_state = asteroids_state.missile_states[0]
        missile_2_state = asteroids_state.missile_states[1]

        # update missile positions
        missile_1_state = jax.lax.cond(
            missile_1_state[5] > 0,
            lambda: jnp.array([self.final_pos(self.consts.MIN_ENTITY_X, self.consts.MAX_ENTITY_X, missile_1_state[0] + missile_1_state[2]), 
                                 self.final_pos(self.consts.MIN_ENTITY_Y, self.consts.MAX_ENTITY_Y, missile_1_state[1] + missile_1_state[3]), 
                                 missile_1_state[2], missile_1_state[3], missile_1_state[4], missile_1_state[5]-1]),
            lambda: jnp.array([0, 0, 0, 0, 0, 0])
        )

        missile_2_state = jax.lax.cond(
            missile_2_state[5] > 0,
            lambda: jnp.array([self.final_pos(self.consts.MIN_ENTITY_X, self.consts.MAX_ENTITY_X, missile_2_state[0] + missile_2_state[2]), 
                                 self.final_pos(self.consts.MIN_ENTITY_Y, self.consts.MAX_ENTITY_Y, missile_2_state[1] + missile_2_state[3]),
                                 missile_2_state[2], missile_2_state[3], missile_2_state[4], missile_2_state[5]-1]),
            lambda: jnp.array([0, 0, 0, 0, 0, 0])
        )

        # initialize missiles on shot
        init_1 = jnp.logical_and(jnp.logical_and(missile_1_state[5] == 0, space), missile_rdy)
        init_2 = jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_not(init_1), missile_2_state[5] == 0), space), missile_rdy)
        missile_1_state = jax.lax.cond(
            jnp.logical_and(init_1, player_active),
            lambda: jnp.array([self.final_pos(self.consts.MIN_ENTITY_X,
                                                self.consts.MAX_ENTITY_X,
                                                self.to_screen_pos(cur_player_x) + self.consts.MISSILE_OFFSET_PER_ROTATION[cur_player_rotation][0]),
                                 self.final_pos(self.consts.MIN_ENTITY_Y,
                                                self.consts.MAX_ENTITY_Y,
                                                self.to_screen_pos(cur_player_y) + self.consts.MISSILE_OFFSET_PER_ROTATION[cur_player_rotation][1]),
                                self.to_screen_pos(self.speed_func(player_speed_x)) + self.consts.MISSILE_SPEED_PER_ROTATION[cur_player_rotation][0],
                                self.to_screen_pos(self.speed_func(player_speed_y)) + self.consts.MISSILE_SPEED_PER_ROTATION[cur_player_rotation][1],
                                cur_player_rotation, self.consts.MISSILE_LIFESPAN]),
            lambda: missile_1_state
        )

        missile_2_state = jax.lax.cond(
            jnp.logical_and(init_2, player_active),
            lambda: jnp.array([self.final_pos(self.consts.MIN_ENTITY_X,
                                                self.consts.MAX_ENTITY_X,
                                                self.to_screen_pos(cur_player_x) + self.consts.MISSILE_OFFSET_PER_ROTATION[cur_player_rotation][0]),
                                 self.final_pos(self.consts.MIN_ENTITY_Y,
                                                self.consts.MAX_ENTITY_Y,
                                                self.to_screen_pos(cur_player_y) + self.consts.MISSILE_OFFSET_PER_ROTATION[cur_player_rotation][1]),
                                self.to_screen_pos(self.speed_func(player_speed_x)) + self.consts.MISSILE_SPEED_PER_ROTATION[cur_player_rotation][0],
                                self.to_screen_pos(self.speed_func(player_speed_y)) + self.consts.MISSILE_SPEED_PER_ROTATION[cur_player_rotation][1],
                                cur_player_rotation, self.consts.MISSILE_LIFESPAN]),
            lambda: missile_2_state
        )

        missile_rdy = jnp.logical_and(jnp.logical_not(jnp.logical_or(init_1, init_2)), jnp.logical_not(space))

        return jnp.array([missile_1_state, missile_2_state]), missile_rdy

    @partial(jax.jit, static_argnums=(0,))
    def asteroids_step(self, asteroids_state: AsteroidsState):
        asteroid_states = asteroids_state.asteroid_states
        side_step_counter = asteroids_state.side_step_counter

        rng_key, subkey = jax.random.split(asteroids_state.rng_key)
        counter_step = jax.random.randint(subkey, [], 7, 10)

        # perform movement on the x-axis if side_step_counter reaches 0
        side_step = jnp.logical_and(side_step_counter <= counter_step, side_step_counter != 0)
        side_step_counter = jax.lax.cond(
            side_step_counter < counter_step,
            lambda: 115 + side_step_counter - counter_step,
            lambda: side_step_counter - counter_step
        )

        @jax.jit
        def update_asteroid(i, asteroid_states):
            ret = jnp.copy(asteroid_states)
            axis_directions = jax.lax.switch(
                ret[i][2],
                [
                    lambda: (self.consts.ASTEROID_SPEED[0], self.consts.ASTEROID_SPEED[1]),
                    lambda: (-self.consts.ASTEROID_SPEED[0], self.consts.ASTEROID_SPEED[1]),
                    lambda: (-self.consts.ASTEROID_SPEED[0], -self.consts.ASTEROID_SPEED[1]),
                    lambda: (self.consts.ASTEROID_SPEED[0], -self.consts.ASTEROID_SPEED[1])
                ]
            )
            return ret.at[i].set(jax.lax.cond(
                ret[i][3] != self.consts.INACTIVE,
                lambda: jnp.array([self.final_pos(self.consts.MIN_ENTITY_X,
                                                  self.consts.MAX_ENTITY_X,
                                                  jax.lax.cond(
                                                      side_step,
                                                      lambda: ret[i][0] + axis_directions[0],
                                                      lambda: ret[i][0])),
                                   self.final_pos(self.consts.MIN_ENTITY_Y,
                                                  self.consts.MAX_ENTITY_Y,
                                                  ret[i][1] + axis_directions[1]),
                                   ret[i][2], ret[i][3], ret[i][4]]),
                lambda: ret[i]
            ))

        # update asteroid positions
        asteroid_states = jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ASTEROIDS, update_asteroid, asteroid_states)

        return asteroid_states, side_step_counter, rng_key

    @partial(jax.jit, static_argnums=(0,))
    def entities_collide(
        self, 
        e1_x: chex.Array,
        e1_y: chex.Array,
        e1_w: chex.Array,
        e1_h: chex.Array,
        e2_x: chex.Array,
        e2_y: chex.Array,
        e2_w: chex.Array,
        e2_h: chex.Array,
    ) -> chex.Array:
        """
        Checks whether two entities are colliding
        """
        overlap_start_x = jnp.maximum(e1_x, e2_x)
        overlap_end_x = jnp.minimum(e1_x + e1_w, e2_x + e2_w)
        overlap_start_y = jnp.maximum(e1_y, e2_y)
        overlap_end_y = jnp.minimum(e1_y + e1_h, e2_y + e2_h)

        return jnp.logical_and(overlap_start_x < overlap_end_x, overlap_start_y < overlap_end_y)

    @partial(jax.jit, static_argnums=(0,))
    def player_hit(self, player_x, player_y, player_active, asteroid_states) -> int:
        """
        Returns index of asteroid hit by player (or -1 if no asteroid is hit)
        """
        @jax.jit
        def get_hit_asteroid(i, index):
            asteroid_dims = jax.lax.switch(
                asteroid_states[i][3],
                [
                    lambda: (0, 0),
                    lambda: self.consts.ASTEROID_SIZE_L,
                    lambda: self.consts.ASTEROID_SIZE_L,
                    lambda: self.consts.ASTEROID_SIZE_M,
                    lambda: self.consts.ASTEROID_SIZE_S
                ]
            )
            # return i if collision with asteroids at index i occurred, old index otherwise
            return jax.lax.cond(
                self.entities_collide(player_x, player_y,
                                      self.consts.PLAYER_SIZE[0], self.consts.PLAYER_SIZE[1],
                                      asteroid_states[i][0], asteroid_states[i][1],
                                      asteroid_dims[0], asteroid_dims[1]),
                lambda: i,
                lambda: index
            )

        return jax.lax.cond(
            player_active,
            lambda: jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ASTEROIDS, get_hit_asteroid, -1),
            lambda: -1
        )

    @partial(jax.jit, static_argnums=(0,))
    def missile_hit(
        self,
        missile_1_x,
        missile_1_y,
        missile_1_active,
        missile_2_x,
        missile_2_y,
        missile_2_active,
        asteroid_states
    ):
        """
        Returns indices of asteroids hit by missiles (or -1 if no asteroid is hit)
        """
        @jax.jit
        def get_hit_asteroid(missile_x, missile_y, i, index):
            asteroid_dims = jax.lax.switch(
                asteroid_states[i][3],
                [
                    lambda: (0, 0),
                    lambda: self.consts.ASTEROID_SIZE_L,
                    lambda: self.consts.ASTEROID_SIZE_L,
                    lambda: self.consts.ASTEROID_SIZE_M,
                    lambda: self.consts.ASTEROID_SIZE_S
                ]
            )
            # return i if collision with asteroids at index i occurred, old index otherwise
            return jax.lax.cond(
                self.entities_collide(missile_x, missile_y,
                                      self.consts.MISSILE_SIZE[0], self.consts.MISSILE_SIZE[1],
                                      asteroid_states[i][0], asteroid_states[i][1],
                                      asteroid_dims[0], asteroid_dims[1]),
                lambda: i,
                lambda: index)

        missile_1_hit = jax.lax.cond(
            missile_1_active,
            lambda: jax.lax.fori_loop(0, asteroid_states.shape[0],
                        lambda i, index: get_hit_asteroid(missile_1_x, missile_1_y, i, index), -1),
            lambda: -1
        )
        missile_2_hit = jax.lax.cond(
            missile_2_active,
            lambda: jax.lax.fori_loop(0, asteroid_states.shape[0],
                        lambda i, index: get_hit_asteroid(missile_2_x, missile_2_y, i, index), -1),
            lambda: -1
        )

        return missile_1_hit, missile_2_hit

    @partial(jax.jit, static_argnums=(0,))
    def destroy_asteroid(self, asteroid_states, index, side_step, rng_key: jax.random.PRNGKey):
        """
        Removes destroyed asteroids from asteroid_states and returns generated score
        """
        asteroid_count = jnp.count_nonzero(asteroid_states, 0)[3]

        # get size of new asteroid(s)
        new_asteroid_size = jax.lax.switch(
            asteroid_states[index][3],
            (
                lambda: self.consts.INACTIVE,
                lambda: self.consts.MEDIUM,
                lambda: self.consts.MEDIUM,
                lambda: self.consts.SMALL,
                lambda: self.consts.INACTIVE
            )
        )

        # get score according to size of destroyed asteroid
        score = jax.lax.switch(
            asteroid_states[index][3],
            (
                lambda: 0,
                lambda: 20,
                lambda: 20,
                lambda: 50,
                lambda: 100
            )
        )

        rng_key, subkey = jax.random.split(rng_key)
        flip_x = jax.random.randint(subkey, [], 0, 4)

        rng_key, subkey = jax.random.split(rng_key)

        # get new asteroid direction
        asteroid_rot = jax.lax.cond( # flip direction with 50% chance
            flip_x % 2 == 1,
            lambda: jnp.bitwise_xor(asteroid_states[index][2], 1),
            lambda: asteroid_states[index][2]
        )
        x_direction = jax.lax.switch(
                asteroid_rot,
                [
                    lambda: self.consts.ASTEROID_SPEED[0],
                    lambda: -self.consts.ASTEROID_SPEED[0],
                    lambda: -self.consts.ASTEROID_SPEED[0],
                    lambda: self.consts.ASTEROID_SPEED[0]
                ]
            )

        # get new asteroid x-position
        asteroid_x = jax.lax.cond(
            side_step,
            lambda: asteroid_states[index][0] + x_direction,
            lambda: asteroid_states[index][0] # same x postition
        )

        # set new asteroid
        new_asteroid_states = asteroid_states.at[index].set(jnp.array([
            asteroid_x,
            asteroid_states[index][1], # same y position
            asteroid_rot,
            new_asteroid_size,
            jax.random.randint(rng_key, [], 0, 8)])) # random color

        # add second new medium asteroid for destruction of large asteroids
        rng_key, subkey = jax.random.split(rng_key)

        # get additional asteroid direction
        asteroid_rot = jax.lax.cond( # flip direction with 50% chance
            flip_x >= 2,
            lambda: jnp.bitwise_xor(asteroid_states[index][2], 1),
            lambda: asteroid_states[index][2]
        )
        x_direction = jax.lax.switch(
                asteroid_rot,
                [
                    lambda: self.consts.ASTEROID_SPEED[0],
                    lambda: -self.consts.ASTEROID_SPEED[0],
                    lambda: -self.consts.ASTEROID_SPEED[0],
                    lambda: self.consts.ASTEROID_SPEED[0]
                ]
            )

        # get additional asteroid x-position
        asteroid_x = jax.lax.cond(
            side_step,
            lambda: asteroid_states[index][0] + x_direction,
            lambda: asteroid_states[index][0] # same x position
        )

        # set additional asteroid if applicable
        new_asteroid_states = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(
                    asteroid_count < self.consts.MAX_NUMBER_OF_ASTEROIDS,
                    new_asteroid_size != self.consts.SMALL),
                new_asteroid_size != self.consts.INACTIVE),
            lambda: new_asteroid_states.at[asteroid_count].set(jnp.array([
                asteroid_x,
                asteroid_states[index][1] + 20,  # y position for second asteroid is increased by 20
                asteroid_rot,
                new_asteroid_size,
                jax.random.randint(rng_key, [], 0, 8)])), # random color
            lambda: new_asteroid_states,
        )

        return new_asteroid_states, score, rng_key

    @partial(jax.jit, static_argnums=(0,))
    def new_stage(self, player_x, player_y, rng_key: jax.random.PRNGKey):
        """
        Generates new asteroids when a stage is cleared
        """
        def get_new_asteroid_state(i, asteroid_states_and_key):
            key, subkey_x = jax.random.split(asteroid_states_and_key[1])
            key, subkey_y = jax.random.split(key)
            key, subkey_rot = jax.random.split(key)
            key, subkey_color = jax.random.split(key)
            key, subkey_shape = jax.random.split(key)

            # generate x-coordinate for a new asteroid such that the central area of the frame stays clear
            asteroid_x = jax.random.randint(subkey_x, [], self.consts.MIN_ENTITY_X,
                self.consts.ASTEROID_BORDER_LEFT + self.consts.MAX_ENTITY_X - self.consts.ASTEROID_BORDER_RIGHT + 1)
            asteroid_x = jax.lax.cond(
                asteroid_x > self.consts.ASTEROID_BORDER_LEFT,
                lambda: self.consts.ASTEROID_BORDER_RIGHT - self.consts.ASTEROID_BORDER_LEFT + asteroid_x,
                lambda: asteroid_x
            )

            # compute whether the new asteroid overlaps with a safe zone around the player on the x-axis
            overlap_x = (jnp.maximum(self.to_screen_pos(player_x) - self.consts.SAFE_ZONE[0], asteroid_x)
                         < jnp.minimum(
                             self.to_screen_pos(player_x) + self.consts.PLAYER_SIZE[0] + self.consts.SAFE_ZONE[0],
                             asteroid_x + self.consts.ASTEROID_SIZE_L[0]))
            
            # compute how much of a safe zone around the player is in frame on the y-axis
            sz_overlap = (jnp.minimum(self.to_screen_pos(player_y) + self.consts.PLAYER_SIZE[1] + self.consts.SAFE_ZONE[1], self.consts.MAX_ENTITY_Y)
                          - jnp.maximum(self.to_screen_pos(player_y) - self.consts.SAFE_ZONE[1], self.consts.MIN_ENTITY_Y))

            # generate preliminary y-coordinate for a new asteroid
            asteroid_y = jax.lax.cond(
                overlap_x,
                lambda: jax.random.randint(subkey_y, [], self.consts.MIN_ENTITY_Y, self.consts.MAX_ENTITY_Y-sz_overlap+1),
                lambda: jax.random.randint(subkey_y, [], self.consts.MIN_ENTITY_Y, self.consts.MAX_ENTITY_Y+1)
            )
            # compute whether the new asteroid overlaps with a safe zone around the player on the y-axis
            overlap_y = (jnp.maximum(self.to_screen_pos(player_y) - self.consts.SAFE_ZONE[1], asteroid_y)
                         < jnp.minimum(self.to_screen_pos(player_y) + self.consts.PLAYER_SIZE[1] + self.consts.SAFE_ZONE[1], asteroid_y + self.consts.ASTEROID_SIZE_L[1]))

            #compute final y-coordinate such that the new asteroid does not overlap with a safe zone around the player
            asteroid_y = jax.lax.cond(
                jnp.logical_and(overlap_x, overlap_y),
                lambda: asteroid_y+sz_overlap,
                lambda: asteroid_y
            )

            # set new asteroid
            return (asteroid_states_and_key[0].at[i].set(jnp.array([
                asteroid_x,
                asteroid_y,
                jax.random.randint(subkey_rot, [], 0, 4),
                jax.random.randint(subkey_shape, [], 1, 3),
                jax.random.randint(subkey_color, [], 0, 8)])),
                key)

        return jax.lax.fori_loop(0, self.consts.NEW_ASTEROIDS_COUNT, get_new_asteroid_state, (jnp.zeros((self.consts.MAX_NUMBER_OF_ASTEROIDS, 5)).astype(jnp.int32), rng_key))

    @partial(jax.jit, static_argnums=(0,))
    def player_safe(self, player_x, player_y, asteroid_states):
        """
        Returns True if no asteroid is within the safe zone around the player
        """
        def get_index_safe(i, safe):
            asteroid_dims = jax.lax.switch(
                asteroid_states[i][3],
                [
                    lambda: (0, 0),
                    lambda: self.consts.ASTEROID_SIZE_L,
                    lambda: self.consts.ASTEROID_SIZE_L,
                    lambda: self.consts.ASTEROID_SIZE_M,
                    lambda: self.consts.ASTEROID_SIZE_S
                ]
            )
            # returns True if player is safe from asteroid i and was safe previously
            return jnp.logical_and(safe,
                jnp.logical_not(
                    self.entities_collide(
                        self.to_screen_pos(player_x) - self.consts.SAFE_ZONE[0],
                        self.to_screen_pos(player_y) - self.consts.SAFE_ZONE[1],
                        self.consts.PLAYER_SIZE[0] + 2*self.consts.SAFE_ZONE[0],
                        self.consts.PLAYER_SIZE[1] + 2*self.consts.SAFE_ZONE[1],
                        asteroid_states[i][0], asteroid_states[i][1], asteroid_dims[0], asteroid_dims[1])))

        return jax.lax.fori_loop(
            0,
            self.consts.MAX_NUMBER_OF_ASTEROIDS,
            get_index_safe,
            True
        )


    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(1234)) -> Tuple[AsteroidsObservation, AsteroidsState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and respective observation.
        """
        state = AsteroidsState(
            player_x=jnp.array(self.consts.INITIAL_PLAYER_X).astype(jnp.int32),
            player_y=jnp.array(self.consts.INITIAL_PLAYER_Y).astype(jnp.int32),
            player_speed_x=jnp.array(0).astype(jnp.int32),
            player_speed_y=jnp.array(0).astype(jnp.int32),
            player_rotation=jnp.array(self.consts.INITIAL_PLAYER_ROTATION).astype(jnp.int32),

            missile_states=jnp.zeros((2, 6)).astype(jnp.int32),
            asteroid_states=jnp.array(self.consts.INITIAL_ASTEROID_STATES).astype(jnp.int32),

            missile_rdy=jnp.array(True).astype(jnp.bool),
            colliding_asteroids=jnp.zeros((3, 3)).astype(jnp.int32),

            score=jnp.array(0).astype(jnp.int32),
            lives=jnp.array(self.consts.STARTING_LIVES).astype(jnp.int32),
            respawn_timer=jnp.array(0).astype(jnp.int32),

            side_step_counter=jnp.array(115).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            rng_key = key
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: AsteroidsState, action: chex.Array
    ) -> Tuple[AsteroidsObservation, AsteroidsState, float, bool, AsteroidsInfo]:

        # update player position, speed and rotation
        player_x, player_y, player_speed_x, player_speed_y, player_rotation, respawn_timer, rng_key = self.player_step(
            state.player_x, state.player_y, state.player_speed_x, state.player_speed_y,
            state.player_rotation, action, state.respawn_timer, state.rng_key
        )
        # player rotation is updated in every fourth step only
        player_rotation = jax.lax.cond(
            (state.step_counter + 2) % 4 == 0,
            lambda: player_rotation,
            lambda: state.player_rotation
        )

        # update missile positions, rotations and lifespans
        missile_states, missile_rdy = self.missile_step(
            state, player_x, player_y, player_speed_x, player_speed_y, player_rotation, action, respawn_timer==0
        )

        # update asteroids
        asteroid_states, side_step_counter, rng_key = self.asteroids_step(state)

        # get all colliding asteroids
        p_hit = self.player_hit(self.to_screen_pos(player_x), self.to_screen_pos(player_y), respawn_timer==0, asteroid_states)
        m_hit = self.missile_hit(missile_states[0][0], missile_states[0][1],
                                 missile_states[0][5] > 0, missile_states[1][0],
                                 missile_states[1][1], missile_states[1][5] > 0,
                                 asteroid_states)
        side_step = side_step_counter > state.side_step_counter

        # update asteroids and score after collisions
        def update_collisions(i, vals):
            index = jax.lax.switch(
                i,
                [
                    lambda: p_hit,
                    lambda: m_hit[0],
                    lambda: m_hit[1],
                ]
            )

            # add asteroid to colliding asteroids
            new_rng_key, sub_key = jax.random.split(vals[3])
            colliding_asteroid = jnp.array([index, vals[0][index][3], jax.random.randint(sub_key, [], 0, 2)])
            new_colliding_asteroids = vals[1].at[i].set(colliding_asteroid)

            # destroy asteroid on hit. DESTRUCTION!!!!!
            new_asteroid_states, add_score, new_rng_key = jax.lax.cond(
                index >= 0,
                lambda: self.destroy_asteroid(vals[0], index, side_step, new_rng_key),
                lambda: (vals[0], 0, new_rng_key)
            )
            return new_asteroid_states, new_colliding_asteroids, vals[2] + add_score, new_rng_key

        colliding_asteroids = jnp.array([[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]])
        asteroid_states, colliding_asteroids, score, rng_key = jax.lax.fori_loop(
            0, 3, update_collisions, (asteroid_states, colliding_asteroids, state.score, rng_key)
        )

        # reset score if it exceeds maximum
        score = jax.lax.cond(
            score >= self.consts.MAX_SCORE,
            lambda: score - self.consts.MAX_SCORE,
            lambda: score
        )

        # set missiles to inactive if they hit an asteroid
        missile_states = jax.lax.cond(
            m_hit[0] >= 0,
            lambda: missile_states.at[0].set(jnp.array([0, 0, 0, 0, 0, 0])),
            lambda: missile_states
        )
        missile_states = jax.lax.cond(
            m_hit[1] >= 0,
            lambda: missile_states.at[1].set(jnp.array([0, 0, 0, 0, 0, 0])),
            lambda: missile_states
        )

        # reset player if hit
        player_x, player_y, player_speed_x, player_speed_y, player_rotation, lives = jax.lax.cond(
            respawn_timer == self.consts.RESPAWN_DELAY - 11,
            lambda: (self.consts.INITIAL_PLAYER_X, self.consts.INITIAL_PLAYER_Y, 0, 0, self.consts.INITIAL_PLAYER_ROTATION, state.lives-1),
            lambda: (player_x, player_y, player_speed_x, player_speed_y, player_rotation, state.lives)
        )

        # update respawn timer if it is not zero already, delay respawn until spawn area is clear
        respawn_timer = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(respawn_timer > 0, respawn_timer <= self.consts.RESPAWN_DELAY),
                jnp.logical_or(
                    self.player_safe(player_x, player_y, asteroid_states),
                    jnp.logical_not(respawn_timer == 1))),
            lambda: respawn_timer-1,
            lambda: respawn_timer
        )
        respawn_timer = jax.lax.cond(
            jnp.logical_and(respawn_timer > self.consts.RESPAWN_DELAY, respawn_timer <= self.consts.RESPAWN_DELAY + self.consts.H_SPACE_DELAY),
            lambda: respawn_timer+1,
            lambda: respawn_timer
        )
        respawn_timer = jax.lax.cond(
            respawn_timer > self.consts.RESPAWN_DELAY + self.consts.H_SPACE_DELAY,
            lambda: 0,
            lambda: respawn_timer
        )
        # set respawn timer when player is hit
        respawn_timer = jax.lax.cond(
            p_hit >= 0,
            lambda: self.consts.RESPAWN_DELAY,
            lambda: respawn_timer
        )

        # update lives
        lives = lives + score//self.consts.POINTS_PER_LIFE - state.score//self.consts.POINTS_PER_LIFE
        # limit lives at maximum
        lives = jax.lax.cond(
            lives > self.consts.MAX_LIVES,
            lambda: self.consts.MAX_LIVES,
            lambda: lives
        )

        # enter new stage if there are no active asteroids left
        asteroid_states, rng_key = jax.lax.cond(
            jnp.count_nonzero(asteroid_states, 0)[3] <= 0,
            lambda: self.new_stage(player_x, player_y, rng_key),
            lambda: (asteroid_states, rng_key)
        )

        # update step counter (reset if lives are zero)
        step_counter = jax.lax.cond(
            lives <= 0,
            lambda: 0,
            lambda: state.step_counter + 1
        )

        # update game state
        new_state = AsteroidsState(
            player_x=player_x,
            player_y=player_y,
            player_speed_x=player_speed_x,
            player_speed_y=player_speed_y,
            player_rotation=player_rotation,

            missile_states=missile_states,
            asteroid_states=asteroid_states,

            missile_rdy=missile_rdy,
            colliding_asteroids=colliding_asteroids,

            lives=lives,
            score=score,
            respawn_timer=respawn_timer,

            side_step_counter=side_step_counter,
            step_counter=step_counter,
            rng_key=rng_key
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_rewards(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def render(self, state: AsteroidsState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AsteroidsState):
        # player
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
            rotation=state.player_rotation,
            active=state.respawn_timer <= 0
        )

        # missiles
        def convert_missile_states_to_entity(missile_states):
            return jnp.array([
                missile_states[0],  # x position
                missile_states[1],  # y position
                self.consts.MISSILE_SIZE[0],  # width
                self.consts.MISSILE_SIZE[1],  # height
                missile_states[4],  # rotation
                missile_states[5] > 0  # active flag
            ])

        missiles = jax.vmap(convert_missile_states_to_entity)(
            state.missile_states
        )

        # asteroids
        def convert_asteroid_states_to_entity(asteroid_states):
            width, height = jax.lax.switch(
                asteroid_states[3],
                [
                    lambda: (0, 0),
                    lambda: (self.consts.ASTEROID_SIZE_L[0], self.consts.ASTEROID_SIZE_L[1]),
                    lambda: (self.consts.ASTEROID_SIZE_L[0], self.consts.ASTEROID_SIZE_L[1]),
                    lambda: (self.consts.ASTEROID_SIZE_M[0], self.consts.ASTEROID_SIZE_M[1]),
                    lambda: (self.consts.ASTEROID_SIZE_S[0], self.consts.ASTEROID_SIZE_S[1])
                ]
            )
            return jnp.array([
                asteroid_states[0],  # x position
                asteroid_states[1],  # y position
                width,  # width
                height,  # height
                asteroid_states[2], # rotation
                asteroid_states[3] != self.consts.INACTIVE  # active flag
            ])

        asteroids = jax.vmap(convert_asteroid_states_to_entity)(
            state.asteroid_states
        )

        return AsteroidsObservation(
            player=player,
            missiles=missiles,
            asteroids=asteroids,

            score=state.score,
            lives=state.lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: AsteroidsObservation) -> jnp.ndarray:
        """Converts the observation to a flat array."""
        return jnp.concatenate([
            jnp.concatenate([
                jnp.atleast_1d(obs.player.x),
                jnp.atleast_1d(obs.player.y),
                jnp.atleast_1d(obs.player.width),
                jnp.atleast_1d(obs.player.height),
                jnp.atleast_1d(obs.player.rotation),
                jnp.atleast_1d(obs.player.active)
            ]),
            obs.missiles.flatten(),
            obs.asteroids.flatten(),

            obs.score.flatten(),
            obs.lives.flatten()
        ]
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Box:
        """Returns the observation space for Asteroids.
        The observation contains:
        - player: EntityPosition (x, y, width, height, rotation, active)
        - missiles: array of shape (2, 6) with (x, y, width, height, rotation, active)
        - asteroids: array of shape (17, 6) with (x, y, width, height, rotation, active)
        - score: int (0-100000)
        - lives: int (0-9)
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160*256, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210*256, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "rotation": spaces.Box(low=0, high=16, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "missiles": spaces.Box(low=0, high=160, shape=(2, 6), dtype=jnp.int32),
            "asteroids": spaces.Box(low=0, high=160, shape=(self.consts.MAX_NUMBER_OF_ASTEROIDS, 6), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=self.consts.MAX_SCORE, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.MAX_LIVES, shape=(), dtype=jnp.int32)
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for Asteroids.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AsteroidsState, all_rewards: chex.Array = None) -> AsteroidsInfo:
        return AsteroidsInfo(score=state.score, step_counter=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsteroidsState, state: AsteroidsState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: AsteroidsState, state: AsteroidsState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AsteroidsState) -> bool:
        return state.lives <= 0

class AsteroidsRenderer(JAXGameRenderer):
    """JAX-based Asteroids game renderer, optimized with JIT compilation."""

    def __init__(self, consts: AsteroidsConstants = None):
        """
        Initializes the renderer by loading sprites, including background.
        """
        super().__init__()
        self.consts = consts or AsteroidsConstants()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/asteroids"
        self.sprites = self._load_sprites()
        # store background sprite directly for use in render function
        self.background = self.sprites.get('background')

    def _load_sprites(self) -> dict[str, Any]:
        """Loads all necessary sprites from .npy files."""
        sprites: Dict[str, Any] = {}

        # helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = jr.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)

        # background
        sprites['background'] = _load_sprite_frame('background')

        # living player
        player_sprites = []
        for i in range(16):
            player_sprites.append(_load_sprite_frame(f'player_pos{i}'))

        # player death animation
        for i in range(3):
            player_sprites.append(_load_sprite_frame(f'death_player{i}'))

        # pad the player and death animation sprites since they have to be used interchangeably 
        # (and jax enforces same sizes)
        player_sprites, _ = jr.pad_to_match(player_sprites)

        for i in range(16):
            sprites[f'player_pos{i}'] = player_sprites[i]
        for i in range(16, 19):
            sprites[f'death_player{i-16}'] = player_sprites[i]

        # digits for score and lives
        digit_path = os.path.join(self.sprite_path, '{}.npy')
        digit_sprites = jr.load_and_pad_digits(digit_path , num_chars=10)
        sprites['digits'] = digit_sprites

        # missiles
        sprites['missile1'] = _load_sprite_frame('missile1')
        sprites['missile2'] = _load_sprite_frame('missile2')

        # asteroids
        asteroid_sprites = []
        for size in ['big1', 'big2', 'medium', 'small']:
            for color in ['brown', 'grey', 'lightblue', 'lightyellow', 'pink', 'purple', 'red', 'yellow']:
                asteroid_sprites.append(_load_sprite_frame(f'asteroid_{size}_{color}'))

        # asteroids death animation
        for size in ['big', 'medium', 'small']:
            for color in ['pink', 'yellow']:
                asteroid_sprites.append(_load_sprite_frame(f'death_{size}_{color}'))

        # pad the asteroid and death animation sprites since they have to be used interchangeably 
        # (and jax enforces same sizes)
        asteroid_sprites, _ = jr.pad_to_match(asteroid_sprites)

        i = 0
        for size in ['big1', 'big2', 'medium', 'small']:
            for color in ['brown', 'grey', 'lightblue', 'lightyellow', 'pink', 'purple', 'red', 'yellow']:
                sprites[f'asteroid_{size}_{color}'] = asteroid_sprites[i]
                i += 1
        for size in ['big', 'medium', 'small']:
            for color in ['pink', 'yellow']:
                sprites[f'death_{size}_{color}'] = asteroid_sprites[i]
                i += 1

        # expand all sprites
        for key, value in sprites.items():
            if isinstance(value, (list, tuple)):
                sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in value]
            else:
                sprites[key] = jnp.expand_dims(value, axis=0)

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: AsteroidsState) -> chex.Array:
        """
        Renders the current game state using JAX operations.

        Args:
            state: A AsteroidsState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        # empty raster
        raster = jr.create_initial_frame(width=self.consts.WIDTH, height=self.consts.HEIGHT)

        # background
        frame_bg = jr.get_sprite_frame(self.background, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        # set player sprite
        player_sprite = jax.lax.switch(
            state.player_rotation,
            [
                lambda: self.sprites['player_pos0'],
                lambda: self.sprites['player_pos1'],
                lambda: self.sprites['player_pos2'],
                lambda: self.sprites['player_pos3'],
                lambda: self.sprites['player_pos4'],
                lambda: self.sprites['player_pos5'],
                lambda: self.sprites['player_pos6'],
                lambda: self.sprites['player_pos7'],
                lambda: self.sprites['player_pos8'],
                lambda: self.sprites['player_pos9'],
                lambda: self.sprites['player_pos10'],
                lambda: self.sprites['player_pos11'],
                lambda: self.sprites['player_pos12'],
                lambda: self.sprites['player_pos13'],
                lambda: self.sprites['player_pos14'],
                lambda: self.sprites['player_pos15']
            ]
        )

        # use death animation instead if player has just died
        player_sprite = jax.lax.cond(
            jnp.logical_or(state.respawn_timer == 134, state.respawn_timer == 135),
            lambda: self.sprites['death_player0'],
            lambda: player_sprite
        )
        player_sprite = jax.lax.cond(
            jnp.isin(state.respawn_timer, jnp.array(range(130, 134))),
            lambda: self.sprites['death_player1'],
            lambda: player_sprite
        )
        player_sprite = jax.lax.cond(
            jnp.isin(state.respawn_timer, jnp.array(range(126, 130))),
            lambda: self.sprites['death_player2'],
            lambda: player_sprite
        )

        frame_player = jr.get_sprite_frame(player_sprite, state.step_counter)
        # only render player in even steps and if player is active
        raster = jax.lax.cond(
            jnp.logical_and(state.step_counter % 2 == 0,
                            jnp.logical_or(jnp.isin(state.respawn_timer, jnp.array(range(126, 136))),
                                           state.respawn_timer==0)),
            lambda: jr.render_at(
                raster, jnp.sign(state.player_x)*(jnp.abs(state.player_x//256))*2,
                jnp.sign(state.player_y)*(jnp.abs(state.player_y)//256)*2, frame_player), # convert x and y position to screen coordinates again
            lambda: raster
        )

        # set missile sprites
        frame_missile1 = jr.get_sprite_frame(self.sprites['missile1'], state.step_counter)
        frame_missile2 = jr.get_sprite_frame(self.sprites['missile2'], state.step_counter)
        # only render missiles in even steps and if it is active
        raster = jax.lax.cond(
            jnp.logical_and(state.step_counter % 2 == 0, state.missile_states[0][5] > 0),
            lambda: jr.render_at(raster, state.missile_states[0][0], state.missile_states[0][1], frame_missile1),
            lambda: raster
        )
        raster = jax.lax.cond(
            jnp.logical_and(state.step_counter % 2 == 0, state.missile_states[1][5] > 0),
            lambda: jr.render_at(raster, state.missile_states[1][0], state.missile_states[1][1], frame_missile2),
            lambda: raster
        )

        # asteroids
        def render_asteroid(i, raster):

            def _get_asteroid_sprite(asteroid_state):

                def _get_asteroid_sprite_by_size(size, color):
                    sprite = jax.lax.switch(
                        color,
                        [
                            lambda: self.sprites[f'asteroid_{size}_brown'],
                            lambda: self.sprites[f'asteroid_{size}_grey'],
                            lambda: self.sprites[f'asteroid_{size}_lightblue'],
                            lambda: self.sprites[f'asteroid_{size}_lightyellow'],
                            lambda: self.sprites[f'asteroid_{size}_pink'],
                            lambda: self.sprites[f'asteroid_{size}_purple'],
                            lambda: self.sprites[f'asteroid_{size}_red'],
                            lambda: self.sprites[f'asteroid_{size}_yellow']
                        ]
                    )
                    return sprite
                
                sprite = jax.lax.switch(
                             asteroid_state[3],
                             [
                                 lambda: _get_asteroid_sprite_by_size('big1', asteroid_state[4]), # does not matter for inactive asteroids
                                 lambda: _get_asteroid_sprite_by_size('big1', asteroid_state[4]),
                                 lambda: _get_asteroid_sprite_by_size('big2', asteroid_state[4]),
                                 lambda: _get_asteroid_sprite_by_size('medium', asteroid_state[4]),
                                 lambda: _get_asteroid_sprite_by_size('small', asteroid_state[4])
                             ]
                         )
                return sprite

            sprite = _get_asteroid_sprite(state.asteroid_states[i])
            frame = jr.get_sprite_frame(sprite, state.step_counter)
            # only render asteroid in odd steps and if it is active and not colliding
            return jax.lax.cond(
                jnp.logical_and(jnp.logical_and(state.step_counter % 2 == 1, state.asteroid_states[i][3] != self.consts.INACTIVE), 
                                jnp.logical_not(jnp.isin(i, state.colliding_asteroids[:,0]))),
                lambda: jr.render_at(raster, state.asteroid_states[i][0], state.asteroid_states[i][1], frame),
                lambda: raster
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ASTEROIDS, render_asteroid, raster)

        # asteroid death animations
        def render_asteroid_death_animation(i, raster):

            def _get_death_sprite(colliding_asteroid):

                def _get_asteroid_sprite_by_size(size, color):
                    sprite = jax.lax.switch(
                        color,
                        [
                            lambda: self.sprites[f'death_{size}_pink'], 
                            lambda: self.sprites[f'death_{size}_yellow']
                        ]
                    )
                    return sprite

                sprite = jax.lax.switch(
                             colliding_asteroid[1],
                             [
                                 lambda: _get_asteroid_sprite_by_size('big', colliding_asteroid[2]), # does not matter for inactive asteroids
                                 lambda: _get_asteroid_sprite_by_size('big', colliding_asteroid[2]),
                                 lambda: _get_asteroid_sprite_by_size('big', colliding_asteroid[2]),
                                 lambda: _get_asteroid_sprite_by_size('medium', colliding_asteroid[2]),
                                 lambda: _get_asteroid_sprite_by_size('small', colliding_asteroid[2])
                             ]
                         )
                return sprite

            sprite = _get_death_sprite(state.colliding_asteroids[i])
            frame = jr.get_sprite_frame(sprite, state.step_counter)
            # only render asteroid death animation in odd steps and if asteroid is colliding
            return jax.lax.cond(
                jnp.logical_and(state.step_counter % 2 == 1, state.colliding_asteroids[i][0] != -1),
                lambda: jr.render_at(raster, state.asteroid_states[state.colliding_asteroids[i][0]][0],
                                       state.asteroid_states[state.colliding_asteroids[i][0]][1], frame),
                lambda: raster,
            )

        raster = jax.lax.fori_loop(0, 3, render_asteroid_death_animation, raster)

        # walls
        wall_color = jnp.array(self.consts.WALL_COLOR, dtype=jnp.uint8)
        # top wall
        raster = raster.at[0:self.consts.WALL_TOP_HEIGHT, :, :].set(wall_color)
        # bottom wall
        raster = raster.at[self.consts.HEIGHT - self.consts.WALL_BOTTOM_HEIGHT:210, :, :].set(wall_color)

        # numbers
        def _get_number_of_digits(val):
            return jax.lax.cond(val < 10, lambda: 1, lambda: 
                                jax.lax.cond(val < 100, lambda: 2, lambda: 
                                jax.lax.cond(val < 1000, lambda: 3, lambda: 
                                jax.lax.cond(val < 10000, lambda: 4, lambda: 5))))    

        digit_sprites = self.sprites.get('digits', None)
        # score
        score_digits = jr.int_to_digits(state.score, max_digits = 5)
        number_of_additional_score_digits = _get_number_of_digits(state.score) - 1
        raster = jr.render_label_selective(raster, 
                                           68 - 16 * number_of_additional_score_digits, 
                                           5, 
                                           score_digits, 
                                           digit_sprites[0], 
                                           4 - number_of_additional_score_digits, 
                                           number_of_additional_score_digits + 1, 
                                           spacing = 16)
        # lives
        lives_digits = jr.int_to_digits(state.lives, max_digits = 1)
        number_of_additional_lives_digits = _get_number_of_digits(state.lives) - 1
        raster = jr.render_label_selective(raster,
                                           132 - 16 * number_of_additional_lives_digits,
                                           5,
                                           lives_digits,
                                           digit_sprites[0],
                                           4 - number_of_additional_lives_digits,
                                           number_of_additional_lives_digits + 1,
                                           spacing = 16)

        return raster