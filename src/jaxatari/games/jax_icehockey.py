import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import jax.random as jrandom
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class IceHockeyConstants(NamedTuple):
    #initialization in OCAtari:
    #Player at (60, 84), (16, 20),
    # Player at (70, 35), (10, 20),
    # Enemy at (70, 155), (10, 20),
    #  Enemy at (84, 105), (15, 21),
    #  Ball at (79, 114), (2, 2),
    #  EnemyScore at (110, 14), (6, 7), 
    # PlayerScore at (46, 14), (6, 7), 
    # Timer at (65, 5), (30, 7)
    #Screen dimensions
    WIDTH: int = 170
    HEIGHT: int = 210
    
    #Game mechanics
    GAME_DURATION: int = 180  # 3 minutes in seconds (3 * 60)
    SCORE_LIMIT: int = 10     # Game ends at 10 points #IGNORE now add it later
    
    # Rink layout
    RINK_LEFT: int = 40
    RINK_RIGHT: int = 140
    RINK_TOP: int = 30
    RINK_BOTTOM: int = 180
    RINK_WIDTH: int = 100     
    RINK_HEIGHT: int = 120    
    
    # Goal areas
    GOAL_WIDTH: int = 40
    GOAL_DEPTH: int = 15
    GOAL_LEFT: int = 60       # (160 - 40) / 2
    GOAL_RIGHT: int = 100     # GOAL_LEFT + GOAL_WIDTH
    
    # Player properties #CHANGE LATER From OC 
    PLAYER_WIDTH: int = 6
    PLAYER_HEIGHT: int = 12
    PLAYER_SPEED: int = 2
    
    # Puck properties
    PUCK_SIZE: int = 3
    PUCK_SPEED: int = 2
    PUCK_MAX_SPEED: int = 4
    PUCK_FRICTION: float = 0.98  # Decay factor per frame (less friction for continuous movement)
    PUCK_START_X: chex.Array = jnp.array(80)    # Center of rink
    PUCK_START_Y: chex.Array = jnp.array(105)   # Center of rink
    PUCK_INITIAL_VEL_X: chex.Array = jnp.array(1)  # Initial horizontal velocity
    PUCK_INITIAL_VEL_Y: chex.Array = jnp.array(1)  # Initial vertical velocity
    
    #Stick Add Later

    PLAYER1_START_X: int = 82   # Team 1, player 1 
    PLAYER1_START_Y: int = 30   
    PLAYER2_START_X: int = 80   # Team 1, player 2 (center field)
    PLAYER2_START_Y: int = 70   
    PLAYER3_START_X: int = 82   # Team 2, player 1 
    PLAYER3_START_Y: int = 170
    PLAYER4_START_X: int = 80   # Team 2, player 2 (center field)
    PLAYER4_START_Y: int = 110  
    
    
    # UI positions
    SCORE_LEFT_X: int = 40
    SCORE_RIGHT_X: int = 110  
    SCORE_Y: int = 14         
    TIMER_X: int = 70        
    TIMER_Y: int = 5
    
    # Logo position
    LOGO_X: int = 73          # X position for logo
    LOGO_Y: int = 188         # Y position for logo         

    #Physics
    COLLISION_DISTANCE: int = 18   
    STICK_LENGTH: int = 8          
    STICK_WIDTH: int = 2           # Width of player stick
    BOUNCE_DAMPING: float = 0.8    # Energy loss on bounce


# immutable state container
class IceHockeyState(NamedTuple):
    #Player information (4 players)  
    players_x: chex.Array          # Horizontal positions of all 4 players, shape: (4,)
    players_y: chex.Array          # Vertical positions of all 4 players, shape: (4,)
    players_vel_x: chex.Array      # Horizontal velocities of all 4 players, shape: (4,)
    players_vel_y: chex.Array      # Vertical velocities of all 4 players, shape: (4,)
    players_dir: chex.Array        # Direction/orientation of players, shape: (4,) - 0=left, 1=right
    
    #Puck information
    puck_x: chex.Array             # Horizontal position of the puck
    puck_y: chex.Array             # Vertical position of the puck
    puck_vel_x: chex.Array         # Horizontal velocity of the puck
    puck_vel_y: chex.Array         # Vertical velocity of the puck
    
    #Game state
    left_score: chex.Array         # Score of the left team (top side)
    right_score: chex.Array        # Score of the right team (bottom side)
    time_remaining: chex.Array     # Time remaining in the game (seconds)
    step_counter: chex.Array       # Number of steps taken in the game
    
    #Player states
    players_has_puck: chex.Array   # Boolean array indicating which player has puck, shape: (4,)
    last_touch: chex.Array         # Index of the last player to touch the puck (-1 if none)


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class IceHockeyObservation(NamedTuple):
    #Player information
    player1: EntityPosition        # Position and size of player 1 (Team 1, top - players)
    player2: EntityPosition        # Position and size of player 2 (Team 1, top - players)
    player3: EntityPosition        # Position and size of player 3 (Team 2, bottom - enemies)
    player4: EntityPosition        # Position and size of player 4 (Team 2, bottom - enemies)
    
    #Puck information
    puck: EntityPosition           # Position and size of the puck
    
    #Game state
    left_score: jnp.ndarray        # Current score of the left team
    right_score: jnp.ndarray       # Current score of the right team
    time_remaining: jnp.ndarray    # Time remaining in the game


class IceHockeyInfo(NamedTuple):
    time: jnp.ndarray              # Current step counter
    all_rewards: chex.Array        # Array of all reward values


class JaxIcehockey(JaxEnvironment[IceHockeyState, IceHockeyObservation, IceHockeyInfo, IceHockeyConstants]):
    """
    JAX-based Ice Hockey game implementation.
    
    This is a 2v2 ice hockey game where:
    - 4 players (2 per team) compete to score goals
    - Players can move in 8 directions and use their stick to hit the puck
    - The puck has realistic physics with friction and bouncing
    - Game ends when time runs out or a team reaches the score limit
    
    Key Features:
    - Real-time physics simulation
    - Collision detection between players, puck, and boundaries
    - Team-based gameplay with different colored players
    - Score tracking and time management
    """
    
    def __init__(self, consts: IceHockeyConstants = None, reward_funcs: list[callable] = None):
        """
        Initialize the Ice Hockey game environment.
        
        Args:
            consts: Game constants (dimensions, speeds, colors, etc.)
            reward_funcs: Optional list of reward functions for RL training
        """
        # Set default constants if none provided
        consts = consts or IceHockeyConstants()
        super().__init__(consts)
        
        # Initialize the renderer (we'll implement this next)
        self.renderer = IceHockeyRenderer(self.consts)
        
        # Store reward functions for RL training
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        
        # Define available actions for ice hockey
        # Players can move in 8 directions + use stick
        self.action_set = [
            Action.NOOP,           # 0: Do nothing
            Action.UP,             # 1: Move up
            Action.DOWN,           # 2: Move down  
            Action.LEFT,           # 3: Move left
            Action.RIGHT,          # 4: Move right
            Action.UPLEFT,         # 5: Move up-left
            Action.UPRIGHT,        # 6: Move up-right
            Action.DOWNLEFT,       # 7: Move down-left
            Action.DOWNRIGHT,      # 8: Move down-right
            Action.FIRE,           # 9: Use stick to hit puck
            Action.UPFIRE,         # 10: Move up + use stick
            Action.DOWNFIRE,       # 11: Move down + use stick
            Action.LEFTFIRE,       # 12: Move left + use stick
            Action.RIGHTFIRE,      # 13: Move right + use stick
            Action.UPLEFTFIRE,     # 14: Move up-left + use stick
            Action.UPRIGHTFIRE,    # 15: Move up-right + use stick
            Action.DOWNLEFTFIRE,   # 16: Move down-left + use stick
            Action.DOWNRIGHTFIRE,  # 17: Move down-right + use stick
        ]
        
        # Observation size: 4 players (x,y) + puck (x,y) + scores + time = 4*2 + 2 + 2 + 1 = 13
        self.obs_size = 4 * 2 + 2 + 2 + 1  # 13 total values
        
        # Game state tracking
        self.max_players = 4  # 2v2 setup
        self.players_per_team = 2
        
        # Team assignments:
        # Team 1 (top side - players): players 0, 1
        # Team 2 (bottom side - enemies): players 2, 3
        self.team1_players = jnp.array([0, 1])
        self.team2_players = jnp.array([2, 3])
        
        print(f"JaxIcehockey initialized with {len(self.action_set)} actions and obs_size={self.obs_size}")

    def action_space(self) -> spaces.Discrete:
        """
        Returns the action space for the ice hockey game.
        
        Returns:
            Discrete action space with 18 possible actions (0-17)
        """
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces:
        """
        Returns the observation space for the ice hockey game.
        
        Returns:
            Dict space containing positions of all players, puck, scores, and time
        """
        return spaces.Dict({
            "player1": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "player2": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "player3": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "player4": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "puck": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "left_score": spaces.Box(low=0, high=self.consts.SCORE_LIMIT, shape=(), dtype=jnp.int32),
            "right_score": spaces.Box(low=0, high=self.consts.SCORE_LIMIT, shape=(), dtype=jnp.int32),
            "time_remaining": spaces.Box(low=0, high=self.consts.GAME_DURATION, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        """
        Returns the image space for rendering.
        
        Returns:
            Box space representing the game screen (210x160x3 RGB)
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8
        )

    def reset(self, key=None) -> Tuple[IceHockeyObservation, IceHockeyState]:
        """
        Reset the game to initial state.
        
        Args:
            key: JAX random key (not used in deterministic reset)
            
        Returns:
            Tuple of (initial_observation, initial_state)
        """
        # Initialize all players at their starting positions
        players_x = jnp.array([
            self.consts.PLAYER1_START_X,
            self.consts.PLAYER2_START_X, 
            self.consts.PLAYER3_START_X,
            self.consts.PLAYER4_START_X
        ]).astype(jnp.int32)
        
        players_y = jnp.array([
            self.consts.PLAYER1_START_Y,
            self.consts.PLAYER2_START_Y,
            self.consts.PLAYER3_START_Y, 
            self.consts.PLAYER4_START_Y
        ]).astype(jnp.int32)
        
        # All players start with zero velocity
        players_vel_x = jnp.zeros(4, dtype=jnp.int32)
        players_vel_y = jnp.zeros(4, dtype=jnp.int32)
        
        # Players start facing right (direction 1)
        players_dir = jnp.ones(4, dtype=jnp.int32)
        
        # Puck starts at center of rink with initial velocity
        puck_x = self.consts.PUCK_START_X.astype(jnp.int32)
        puck_y = self.consts.PUCK_START_Y.astype(jnp.int32)
        puck_vel_x = self.consts.PUCK_INITIAL_VEL_X.astype(jnp.int32)
        puck_vel_y = self.consts.PUCK_INITIAL_VEL_Y.astype(jnp.int32)
        
        # Game starts with no scores and full time
        left_score = jnp.array(0, dtype=jnp.int32)
        right_score = jnp.array(0, dtype=jnp.int32)
        time_remaining = jnp.array(self.consts.GAME_DURATION, dtype=jnp.int32)
        step_counter = jnp.array(0, dtype=jnp.int32)
        
        # No player has the puck initially
        players_has_puck = jnp.zeros(4, dtype=jnp.int32)
        last_touch = jnp.array(-1, dtype=jnp.int32)  # -1 means no one touched it
        
        # Create initial state
        state = IceHockeyState(
            players_x=players_x,
            players_y=players_y,
            players_vel_x=players_vel_x,
            players_vel_y=players_vel_y,
            players_dir=players_dir,
            puck_x=puck_x,
            puck_y=puck_y,
            puck_vel_x=puck_vel_x,
            puck_vel_y=puck_vel_y,
            left_score=left_score,
            right_score=right_score,
            time_remaining=time_remaining,
            step_counter=step_counter,
            players_has_puck=players_has_puck,
            last_touch=last_touch,
        )
        
        # Get initial observation
        initial_obs = self._get_observation(state)
        
        print(f"Game reset: Players at {players_x}, {players_y}, Puck at ({puck_x}, {puck_y})")
        return initial_obs, state

    def render(self, state: IceHockeyState) -> jnp.ndarray:
        """
        Render the current game state as an image.
        
        Args:
            state: Current game state
            
        Returns:
            RGB image array of shape (height, width, 3)
        """
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: IceHockeyState, action: chex.Array) -> Tuple[IceHockeyObservation, IceHockeyState, chex.Array, bool, IceHockeyInfo]:
        """
        Execute one step of the game.
        
        Args:
            state: Current game state
            action: Action to take (0-17)
            
        Returns:
            Tuple of (observation, new_state, reward, done, info)
        """
        # Convert action to movement and stick usage
        movement, use_stick = self._parse_action(action)
        
        # Update all players: Team 1 (controlled) + Team 2 (AI)
        new_players_x, new_players_y, new_players_vel_x, new_players_vel_y, new_players_dir = self._update_all_players(
            state, movement
        )
        
        # Update puck physics
        new_puck_x, new_puck_y, new_puck_vel_x, new_puck_vel_y, new_players_has_puck, new_last_touch = self._update_puck(
            state, new_players_x, new_players_y, use_stick
        )
        
        # Check for goals and update scores
        new_left_score, new_right_score, goal_scored = self._check_goals(
            state.left_score, state.right_score, new_puck_x, new_puck_y
        )
        
        # Reset all positions if goal was scored (but keep time and scores)
        reset_players_x = jnp.array([
            self.consts.PLAYER1_START_X,
            self.consts.PLAYER2_START_X, 
            self.consts.PLAYER3_START_X,
            self.consts.PLAYER4_START_X
        ]).astype(jnp.int32)
        
        reset_players_y = jnp.array([
            self.consts.PLAYER1_START_Y,
            self.consts.PLAYER2_START_Y,
            self.consts.PLAYER3_START_Y, 
            self.consts.PLAYER4_START_Y
        ]).astype(jnp.int32)
        
        reset_players_vel_x = jnp.zeros(4, dtype=jnp.float32)
        reset_players_vel_y = jnp.zeros(4, dtype=jnp.float32)
        reset_players_dir = jnp.ones(4, dtype=jnp.int32)
        reset_players_has_puck = jnp.zeros(4, dtype=jnp.int32)
        reset_last_touch = jnp.array(-1, dtype=jnp.int32)
        
        # Reset puck and players if goal was scored
        puck_x_final, puck_y_final, puck_vel_x_final, puck_vel_y_final = jax.lax.cond(
            goal_scored,
            lambda: (self.consts.PUCK_START_X.astype(jnp.int32), 
                    self.consts.PUCK_START_Y.astype(jnp.int32),
                    self.consts.PUCK_INITIAL_VEL_X.astype(jnp.int32),
                    self.consts.PUCK_INITIAL_VEL_Y.astype(jnp.int32)),
            lambda: (new_puck_x.astype(jnp.int32), new_puck_y.astype(jnp.int32), 
                    new_puck_vel_x.astype(jnp.int32), new_puck_vel_y.astype(jnp.int32))
        )
        
        players_x_final, players_y_final, players_vel_x_final, players_vel_y_final, players_dir_final, players_has_puck_final, last_touch_final = jax.lax.cond(
            goal_scored,
            lambda: (reset_players_x, reset_players_y, reset_players_vel_x, reset_players_vel_y, reset_players_dir, reset_players_has_puck, reset_last_touch),
            lambda: (new_players_x, new_players_y, new_players_vel_x, new_players_vel_y, new_players_dir, new_players_has_puck, new_last_touch)
        )
        
        # Update time (decrease every 60 steps = 1 second at 60 FPS)
        new_time_remaining = jax.lax.cond(
            state.step_counter % 60 == 0,
            lambda: jnp.maximum(state.time_remaining - 1, 0),
            lambda: state.time_remaining
        )
        
        # Create new state
        new_state = IceHockeyState(
            players_x=players_x_final,
            players_y=players_y_final,
            players_vel_x=players_vel_x_final,
            players_vel_y=players_vel_y_final,
            players_dir=players_dir_final,
            puck_x=puck_x_final,
            puck_y=puck_y_final,
            puck_vel_x=puck_vel_x_final,
            puck_vel_y=puck_vel_y_final,
            left_score=new_left_score,
            right_score=new_right_score,
            time_remaining=new_time_remaining,
            step_counter=state.step_counter + 1,
            players_has_puck=players_has_puck_final,
            last_touch=last_touch_final,
        )
        
        # Calculate reward and check if done
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)
        
        return observation, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: IceHockeyState) -> IceHockeyObservation:
        """
        Convert game state to observation format.
        
        Args:
            state: Current game state
            
        Returns:
            Observation containing player positions, puck position, scores, and time
        """
        # Create EntityPosition objects for each player
        player1 = EntityPosition(
            x=state.players_x[0],
            y=state.players_y[0],
            width=jnp.array(self.consts.PLAYER_WIDTH),
            height=jnp.array(self.consts.PLAYER_HEIGHT),
        )
        
        player2 = EntityPosition(
            x=state.players_x[1],
            y=state.players_y[1],
            width=jnp.array(self.consts.PLAYER_WIDTH),
            height=jnp.array(self.consts.PLAYER_HEIGHT),
        )
        
        player3 = EntityPosition(
            x=state.players_x[2],
            y=state.players_y[2],
            width=jnp.array(self.consts.PLAYER_WIDTH),
            height=jnp.array(self.consts.PLAYER_HEIGHT),
        )
        
        player4 = EntityPosition(
            x=state.players_x[3],
            y=state.players_y[3],
            width=jnp.array(self.consts.PLAYER_WIDTH),
            height=jnp.array(self.consts.PLAYER_HEIGHT),
        )
        
        # Create EntityPosition for puck
        puck = EntityPosition(
            x=state.puck_x,
            y=state.puck_y,
            width=jnp.array(self.consts.PUCK_SIZE),
            height=jnp.array(self.consts.PUCK_SIZE),
        )
        
        return IceHockeyObservation(
            player1=player1,
            player2=player2,
            player3=player3,
            player4=player4,
            puck=puck,
            left_score=state.left_score,
            right_score=state.right_score,
            time_remaining=state.time_remaining,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: IceHockeyObservation) -> jnp.ndarray:
        """
        Convert observation to flat array for ML algorithms.
        
        Args:
            obs: Observation object
            
        Returns:
            Flattened array of all observation values
        """
        return jnp.concatenate([
            obs.player1.x.flatten(),
            obs.player1.y.flatten(),
            obs.player2.x.flatten(),
            obs.player2.y.flatten(),
            obs.player3.x.flatten(),
            obs.player3.y.flatten(),
            obs.player4.x.flatten(),
            obs.player4.y.flatten(),
            obs.puck.x.flatten(),
            obs.puck.y.flatten(),
            obs.left_score.flatten(),
            obs.right_score.flatten(),
            obs.time_remaining.flatten(),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: IceHockeyState, all_rewards: chex.Array) -> IceHockeyInfo:
        """
        Get additional game information.
        
        Args:
            state: Current game state
            all_rewards: Array of reward values
            
        Returns:
            Info object containing step counter and rewards
        """
        return IceHockeyInfo(
            time=state.step_counter,
            all_rewards=all_rewards
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: IceHockeyState, state: IceHockeyState) -> chex.Array:
        """
        Calculate reward for the current step.
        
        Args:
            previous_state: Previous game state
            state: Current game state
            
        Returns:
            Reward value (positive for good actions, negative for bad)
        """
        # Simple reward: +1 for scoring, -1 for being scored against
        score_reward = (state.left_score - state.right_score) - (previous_state.left_score - previous_state.right_score)
        return score_reward.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: IceHockeyState, state: IceHockeyState) -> chex.Array:
        """
        Calculate all reward components.
        
        Args:
            previous_state: Previous game state
            state: Current game state
            
        Returns:
            Array of all reward values
        """
        if self.reward_funcs is None:
            return jnp.zeros(1)
        
        rewards = jnp.array([
            reward_func(previous_state, state) for reward_func in self.reward_funcs
        ])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: IceHockeyState) -> bool:
        """
        Check if the game is finished.
        
        Args:
            state: Current game state
            
        Returns:
            True if game is over, False otherwise
        """
        # Game ends only when time runs out (no score limit)
        time_up = state.time_remaining <= 0
        return time_up

    @partial(jax.jit, static_argnums=(0,))
    def _parse_action(self, action: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Parse action into movement direction and stick usage.
        
        Args:
            action: Action value (0-17)
            
        Returns:
            Tuple of (movement_direction, use_stick)
        """
        # Movement directions: 0=noop, 2=up, 3=left, 4=right, 5=down, 6=up-right, 7=down-left, 8=down-right
        movement = jnp.where(action == 0, 0,  # NOOP
                   jnp.where(action == 1, 0,  #FIRE
                   jnp.where(action == 2, 1,  # UP
                   jnp.where(action == 3, 4,  # LEFT
                   jnp.where(action == 4, 3,  # RIGHT
                   jnp.where(action == 5, 2,  # Down
                   jnp.where(action == 6, 6,  # UPRIGHT
                   jnp.where(action == 7, 7,  # DOWNLEFT
                   jnp.where(action == 8, 8,  # DOWNRIGHT
                   jnp.where(action == 9, 0,  # FIRE (no movement)
                   jnp.where(action == 10, 1, # UPFIRE
                   jnp.where(action == 11, 4, # RIGHTFIRE
                   jnp.where(action == 12, 3, # LEFTFIRE
                   jnp.where(action == 13, 2, # DOWNFIRE
                   jnp.where(action == 14, 5, # UPLEFTFIRE
                   jnp.where(action == 15, 6, # UPRIGHTFIRE
                   jnp.where(action == 16, 7, # DOWNLEFTFIRE
                   jnp.where(action == 17, 8, # DOWNRIGHTFIRE
                   0))))))))))))))))))
        
        # Stick usage: actions 9-17 use stick
        use_stick = jnp.where(action >= 9, 1, 0)
        
        return movement, use_stick

    @partial(jax.jit, static_argnums=(0,))
    def _update_players(self, state: IceHockeyState, movement: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Update player positions, velocities, and directions.
        
        Args:
            state: Current game state
            movement: Movement direction for the controlled player
            
        Returns:
            Tuple of (new_x, new_y, new_vel_x, new_vel_y, new_dir)
        """
        # Calculate velocity changes based on movement
        vel_x_change = jnp.where(movement == 3, -self.consts.PLAYER_SPEED,  # LEFT
                       jnp.where(movement == 4, self.consts.PLAYER_SPEED,   # RIGHT
                       jnp.where(movement == 5, -self.consts.PLAYER_SPEED,  # UPLEFT
                       jnp.where(movement == 6, self.consts.PLAYER_SPEED,   # UPRIGHT
                       jnp.where(movement == 7, -self.consts.PLAYER_SPEED,  # DOWNLEFT
                       jnp.where(movement == 8, self.consts.PLAYER_SPEED,   # DOWNRIGHT
                       0))))))
        
        vel_y_change = jnp.where(movement == 1, -self.consts.PLAYER_SPEED,  # UP
                       jnp.where(movement == 2, self.consts.PLAYER_SPEED,   # DOWN
                       jnp.where(movement == 5, -self.consts.PLAYER_SPEED,  # UPLEFT
                       jnp.where(movement == 6, -self.consts.PLAYER_SPEED,  # UPRIGHT
                       jnp.where(movement == 7, self.consts.PLAYER_SPEED,   # DOWNLEFT
                       jnp.where(movement == 8, self.consts.PLAYER_SPEED,   # DOWNRIGHT
                       0))))))
        
        # Find which player is closest to the puck (Team 1 - players 0,1)
        distances_to_puck = jnp.array([
            jnp.sqrt((state.players_x[0] - state.puck_x)**2 + (state.players_y[0] - state.puck_y)**2),
            jnp.sqrt((state.players_x[1] - state.puck_x)**2 + (state.players_y[1] - state.puck_y)**2)
        ])
        closest_player_idx = jnp.argmin(distances_to_puck)  # 0 or 1
        
        # Only the closest player to the puck moves (Team 1)
        new_vel_x = state.players_vel_x
        new_vel_y = state.players_vel_y
        
        # Update velocity for the closest player only
        new_vel_x = new_vel_x.at[closest_player_idx].set(
            (state.players_vel_x[closest_player_idx] + vel_x_change) * 0.8
        )
        new_vel_y = new_vel_y.at[closest_player_idx].set(
            (state.players_vel_y[closest_player_idx] + vel_y_change) * 0.8
        )
        
        # Apply damping to all players
        new_vel_x = new_vel_x * 0.9
        new_vel_y = new_vel_y * 0.9
        
        # Update positions for all players
        new_x = state.players_x + new_vel_x.astype(jnp.int32)
        new_y = state.players_y + new_vel_y.astype(jnp.int32)
        
        # Keep players within rink boundaries
        new_x = jnp.clip(new_x, 
                        self.consts.RINK_LEFT + self.consts.PLAYER_WIDTH // 2,
                        self.consts.RINK_RIGHT - self.consts.PLAYER_WIDTH // 2)
        new_y = jnp.clip(new_y,
                        self.consts.RINK_TOP + self.consts.PLAYER_HEIGHT // 2,
                        self.consts.RINK_BOTTOM - self.consts.PLAYER_HEIGHT // 2)
        
        # Update direction based on movement (only for the moving player)
        new_dir = state.players_dir
        new_dir = new_dir.at[closest_player_idx].set(
            jnp.where(vel_x_change > 0, 1,  # Facing right
                     jnp.where(vel_x_change < 0, 0,  # Facing left
                     state.players_dir[closest_player_idx]))  # Keep current direction
        )
        
        return new_x, new_y, new_vel_x, new_vel_y, new_dir

    @partial(jax.jit, static_argnums=(0,))
    def _update_all_players(self, state: IceHockeyState, movement: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Update all players: Team 1 (controlled by player) + Team 2 (AI controlled).
        
        Args:
            state: Current game state
            movement: Movement direction for the controlled player
            
        Returns:
            Tuple of (new_x, new_y, new_vel_x, new_vel_y, new_dir)
        """
        # Calculate velocity changes based on movement for Team 1
        vel_x_change = jnp.where(movement == 3, -self.consts.PLAYER_SPEED,  # LEFT
                       jnp.where(movement == 4, self.consts.PLAYER_SPEED,   # RIGHT
                       jnp.where(movement == 5, -self.consts.PLAYER_SPEED,  # UPLEFT
                       jnp.where(movement == 6, self.consts.PLAYER_SPEED,   # UPRIGHT
                       jnp.where(movement == 7, -self.consts.PLAYER_SPEED,  # DOWNLEFT
                       jnp.where(movement == 8, self.consts.PLAYER_SPEED,   # DOWNRIGHT
                       0))))))
        
        vel_y_change = jnp.where(movement == 1, -self.consts.PLAYER_SPEED,  # UP
                       jnp.where(movement == 2, self.consts.PLAYER_SPEED,   # DOWN
                       jnp.where(movement == 5, -self.consts.PLAYER_SPEED,  # UPLEFT
                       jnp.where(movement == 6, -self.consts.PLAYER_SPEED,  # UPRIGHT
                       jnp.where(movement == 7, self.consts.PLAYER_SPEED,   # DOWNLEFT
                       jnp.where(movement == 8, self.consts.PLAYER_SPEED,   # DOWNRIGHT
                       0))))))
        
        # Find which Team 1 player is closest to the puck (players 0, 1)
        team1_distances = jnp.array([
            jnp.sqrt((state.players_x[0] - state.puck_x)**2 + (state.players_y[0] - state.puck_y)**2),
            jnp.sqrt((state.players_x[1] - state.puck_x)**2 + (state.players_y[1] - state.puck_y)**2)
        ])
        closest_team1_idx = jnp.argmin(team1_distances)  # 0 or 1
        
        # Find which Team 2 player is closest to the puck (players 2, 3)
        team2_distances = jnp.array([
            jnp.sqrt((state.players_x[2] - state.puck_x)**2 + (state.players_y[2] - state.puck_y)**2),
            jnp.sqrt((state.players_x[3] - state.puck_x)**2 + (state.players_y[3] - state.puck_y)**2)
        ])
        closest_team2_idx = jnp.argmin(team2_distances) + 2  # 2 or 3
        
        # Calculate AI movement for Team 2 (towards puck)
        dx_to_puck = state.puck_x - state.players_x[closest_team2_idx]
        dy_to_puck = state.puck_y - state.players_y[closest_team2_idx]
        distance_to_puck = jnp.sqrt(dx_to_puck**2 + dy_to_puck**2)
        dx_normalized = jnp.where(distance_to_puck > 0, dx_to_puck / distance_to_puck, 0)
        dy_normalized = jnp.where(distance_to_puck > 0, dy_to_puck / distance_to_puck, 0)
        
        # Initialize new velocities
        new_vel_x = state.players_vel_x
        new_vel_y = state.players_vel_y
        
        # Update Team 1 player (controlled)
        new_vel_x = new_vel_x.at[closest_team1_idx].set(
            (state.players_vel_x[closest_team1_idx] + vel_x_change) * 0.8
        )
        new_vel_y = new_vel_y.at[closest_team1_idx].set(
            (state.players_vel_y[closest_team1_idx] + vel_y_change) * 0.8
        )
        
        # Update Team 2 player (AI)
        new_vel_x = new_vel_x.at[closest_team2_idx].set(
            dx_normalized * self.consts.PLAYER_SPEED
        )
        new_vel_y = new_vel_y.at[closest_team2_idx].set(
            dy_normalized * self.consts.PLAYER_SPEED
        )
        
        # Apply damping to all players
        new_vel_x = new_vel_x * 0.9
        new_vel_y = new_vel_y * 0.9
        
        # Update positions for all players
        new_x = state.players_x + new_vel_x.astype(jnp.int32)
        new_y = state.players_y + new_vel_y.astype(jnp.int32)
        
        # Keep players within rink boundaries
        new_x = jnp.clip(new_x, 
                        self.consts.RINK_LEFT + self.consts.PLAYER_WIDTH // 2,
                        self.consts.RINK_RIGHT - self.consts.PLAYER_WIDTH // 2)
        new_y = jnp.clip(new_y,
                        self.consts.RINK_TOP + self.consts.PLAYER_HEIGHT // 2,
                        self.consts.RINK_BOTTOM - self.consts.PLAYER_HEIGHT // 2)
        
        # Update directions
        new_dir = state.players_dir
        
        # Update direction for Team 1 player
        new_dir = new_dir.at[closest_team1_idx].set(
            jnp.where(vel_x_change > 0, 1,  # Facing right
                     jnp.where(vel_x_change < 0, 0,  # Facing left
                     state.players_dir[closest_team1_idx]))  # Keep current direction
        )
        
        # Update direction for Team 2 player
        new_dir = new_dir.at[closest_team2_idx].set(
            jnp.where(dx_normalized > 0, 1,  # Facing right
                     jnp.where(dx_normalized < 0, 0,  # Facing left
                     state.players_dir[closest_team2_idx]))  # Keep current direction
        )
        
        return new_x, new_y, new_vel_x, new_vel_y, new_dir

    @partial(jax.jit, static_argnums=(0,))
    def _update_enemies(self, state: IceHockeyState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Update enemy AI players (Team 2 - players 2,3).
        They follow the puck with simple AI.
        
        Args:
            state: Current game state
            
        Returns:
            Tuple of (new_x, new_y, new_vel_x, new_vel_y, new_dir)
        """
        # Find which Team 1 player is closest to the puck (players 0, 1)
        team1_distances = jnp.array([
            jnp.sqrt((state.players_x[0] - state.puck_x)**2 + (state.players_y[0] - state.puck_y)**2),
            jnp.sqrt((state.players_x[1] - state.puck_x)**2 + (state.players_y[1] - state.puck_y)**2)
        ])
        active_player_idx = jnp.argmin(team1_distances)  # 0 or 1
        
        # Find which Team 2 player is closest to the puck (players 2, 3)
        team2_distances = jnp.array([
            jnp.sqrt((state.players_x[2] - state.puck_x)**2 + (state.players_y[2] - state.puck_y)**2),
            jnp.sqrt((state.players_x[3] - state.puck_x)**2 + (state.players_y[3] - state.puck_y)**2)
        ])
        active_enemy_idx = jnp.argmin(team2_distances) + 2  # 2 or 3
        
        # Strategic AI: Enemy tries to get puck and guide it to opponent's goal
        # If enemy has puck or is close to it, move towards opponent's goal
        # If player has puck or is close to it, try to intercept and take it
        
        # Check who is closer to the puck (only active players)
        active_player_distance = jnp.sqrt((state.players_x[active_player_idx] - state.puck_x)**2 + (state.players_y[active_player_idx] - state.puck_y)**2)
        active_enemy_distance = jnp.sqrt((state.players_x[active_enemy_idx] - state.puck_x)**2 + (state.players_y[active_enemy_idx] - state.puck_y)**2)
        
        # Determine strategy based on puck possession
        enemy_has_advantage = active_enemy_distance < active_player_distance
        
        # Calculate target position with lateral movement to avoid opponent
        # Get opponent position (closest player from Team 1)
        opponent_x = state.players_x[active_player_idx]
        opponent_y = state.players_y[active_player_idx]
        
        # Calculate lateral offset to move around opponent
        lateral_offset = 20  # Distance to move around opponent
        enemy_x = state.players_x[active_enemy_idx]
        
        # Determine which side to go around opponent
        go_left = enemy_x < opponent_x  # If enemy is left of opponent, go further left
        
        target_x = jnp.where(enemy_has_advantage,
                            # If enemy is closer, move towards player's goal with lateral movement
                            jnp.where(go_left,
                                    (self.consts.GOAL_LEFT + self.consts.GOAL_RIGHT) // 2 - lateral_offset,
                                    (self.consts.GOAL_LEFT + self.consts.GOAL_RIGHT) // 2 + lateral_offset),
                            # If player is closer, intercept the puck with lateral movement
                            jnp.where(go_left,
                                    state.puck_x - lateral_offset,
                                    state.puck_x + lateral_offset))
        
        target_y = jnp.where(enemy_has_advantage,
                            # If enemy is closer, move towards player's goal (top goal)
                            self.consts.RINK_TOP + self.consts.GOAL_DEPTH // 2,
                            # If player is closer, intercept the puck
                            state.puck_y)
        
        # Calculate direction to target
        dx_to_target = target_x - state.players_x[active_enemy_idx]
        dy_to_target = target_y - state.players_y[active_enemy_idx]
        
        # Normalize direction
        distance_to_target = jnp.sqrt(dx_to_target**2 + dy_to_target**2)
        dx_normalized = jnp.where(distance_to_target > 0, dx_to_target / distance_to_target, 0)
        dy_normalized = jnp.where(distance_to_target > 0, dy_to_target / distance_to_target, 0)
        
        # Calculate velocity towards target
        enemy_vel_x = dx_normalized * self.consts.PLAYER_SPEED
        enemy_vel_y = dy_normalized * self.consts.PLAYER_SPEED
        
        # Update velocities for all players
        new_vel_x = state.players_vel_x
        new_vel_y = state.players_vel_y
        
        # Update velocity for Enemy 2 only (the active field player)
        new_vel_x = new_vel_x.at[active_enemy_idx].set(enemy_vel_x)
        new_vel_y = new_vel_y.at[active_enemy_idx].set(enemy_vel_y)
        
        # Apply damping to all players
        new_vel_x = new_vel_x * 0.9
        new_vel_y = new_vel_y * 0.9
        
        # Update positions for all players
        new_x = state.players_x + new_vel_x.astype(jnp.int32)
        new_y = state.players_y + new_vel_y.astype(jnp.int32)
        
        # Keep players within rink boundaries
        new_x = jnp.clip(new_x, 
                        self.consts.RINK_LEFT + self.consts.PLAYER_WIDTH // 2,
                        self.consts.RINK_RIGHT - self.consts.PLAYER_WIDTH // 2)
        new_y = jnp.clip(new_y,
                        self.consts.RINK_TOP + self.consts.PLAYER_HEIGHT // 2,
                        self.consts.RINK_BOTTOM - self.consts.PLAYER_HEIGHT // 2)
        
        # Update direction based on movement (only for the moving enemy)
        new_dir = state.players_dir
        new_dir = new_dir.at[active_enemy_idx].set(
            jnp.where(enemy_vel_x > 0, 1,  # Facing right
                     jnp.where(enemy_vel_x < 0, 0,  # Facing left
                     state.players_dir[active_enemy_idx]))  # Keep current direction
        )
        
        return new_x, new_y, new_vel_x, new_vel_y, new_dir

    @partial(jax.jit, static_argnums=(0,))
    def _update_puck(self, state: IceHockeyState, players_x: chex.Array, players_y: chex.Array, use_stick: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Update puck physics and handle player interactions.
        
        Args:
            state: Current game state
            players_x: New player x positions
            players_y: New player y positions
            use_stick: Whether players are using their sticks
            
        Returns:
            Tuple of (new_puck_x, new_puck_y, new_puck_vel_x, new_puck_vel_y, players_has_puck, last_touch)
        """
        # Apply friction to puck (like Pong ball)
        new_puck_vel_x = (state.puck_vel_x * self.consts.PUCK_FRICTION).astype(jnp.int32)
        new_puck_vel_y = (state.puck_vel_y * self.consts.PUCK_FRICTION).astype(jnp.int32)
        
        # Update puck position
        new_puck_x = state.puck_x + new_puck_vel_x.astype(jnp.int32)
        new_puck_y = state.puck_y + new_puck_vel_y.astype(jnp.int32)
        
        # Check for wall collisions (like Pong)
        wall_bounce_x = jnp.logical_or(new_puck_x <= self.consts.RINK_LEFT,
                                     new_puck_x >= self.consts.RINK_RIGHT)
        wall_bounce_y = jnp.logical_or(new_puck_y <= self.consts.RINK_TOP,
                                     new_puck_y >= self.consts.RINK_BOTTOM)
        
        # Bounce off walls (like Pong ball)
        new_puck_vel_x = jnp.where(wall_bounce_x, -new_puck_vel_x, new_puck_vel_x)
        new_puck_vel_y = jnp.where(wall_bounce_y, -new_puck_vel_y, new_puck_vel_y)
        
        # Keep puck within bounds
        new_puck_x = jnp.clip(new_puck_x, self.consts.RINK_LEFT, self.consts.RINK_RIGHT)
        new_puck_y = jnp.clip(new_puck_y, self.consts.RINK_TOP, self.consts.RINK_BOTTOM)
        
        # Check for player-puck collisions - only closest player can hit the puck
        players_has_puck = jnp.zeros(4, dtype=jnp.int32)
        new_last_touch = state.last_touch
        
        # Calculate distances from all players to puck
        distances = jnp.zeros(4, dtype=jnp.float32)
        for i in range(4):
            dx = new_puck_x - players_x[i]
            dy = new_puck_y - players_y[i]
            distances = distances.at[i].set(jnp.sqrt(dx**2 + dy**2))
        
        # Find the closest player to the puck
        closest_player_idx = jnp.argmin(distances)
        closest_distance = distances[closest_player_idx]
        
        # Check if any Team 1 player (player) is using stick and close enough to steal puck
        team1_distances = distances[:2]  # Players 0, 1
        team1_closest_distance = jnp.min(team1_distances)
        team1_using_stick = jnp.any(use_stick)  # Check if any player is using stick
        
        # Player gets priority if they're using stick and close enough, even if not closest
        player_can_steal = jnp.logical_and(
            team1_using_stick,
            team1_closest_distance <= self.consts.COLLISION_DISTANCE + 8
        )
        
        # Check if enemy can steal from player (50% chance when player uses stick)
        # Only if player is currently controlling the puck and using stick
        enemy_steal_chance = 0.3
        # Use step counter for pseudo-randomness (50% chance)
        pseudo_random = (state.step_counter % 2) / 2.0  # Cycles through 0.0, 0.5
        enemy_can_steal = jnp.logical_and(
            team1_using_stick,  # Player is using stick
            jnp.logical_and(
                closest_distance <= self.consts.COLLISION_DISTANCE + 8,  # Enemy is close enough
                pseudo_random < enemy_steal_chance  # 50% chance (when pseudo_random is 0.0)
            )
        )
        
        # Determine who controls the puck
        # Priority: 1) Enemy stealing from player (when player uses stick), 2) Player stealing from enemy, 3) Closest player
        controlling_player_idx = jnp.where(
            enemy_can_steal,
            closest_player_idx,  # Enemy steals from player
            jnp.where(
                player_can_steal,
                jnp.argmin(team1_distances),  # Closest Team 1 player
                closest_player_idx  # Normal closest player logic
            )
        )
        
        # Only the controlling player can control the puck if they're close enough
        can_control_puck = jnp.where(
            enemy_can_steal,
            closest_distance <= self.consts.COLLISION_DISTANCE + 8,
            jnp.where(
                player_can_steal,
                team1_closest_distance <= self.consts.COLLISION_DISTANCE + 8,
                closest_distance <= self.consts.COLLISION_DISTANCE + 8
            )
        )
        
        # Check if controlling player is from Team 1 (players 0,1) or Team 2 (players 2,3)
        is_team1 = controlling_player_idx < 2
        
        # Get the controlling player's velocity for puck carrying
        controlling_player_vel_x = state.players_vel_x[controlling_player_idx]
        controlling_player_vel_y = state.players_vel_y[controlling_player_idx]
        
        # Calculate puck velocity based on player movement (carrying) vs hitting
        # If player is moving, carry the puck with them; if not moving much, hit toward goal
        player_moving = jnp.abs(controlling_player_vel_x) + jnp.abs(controlling_player_vel_y) > 0.5
        
        # Puck carrying: move puck with player's velocity
        carry_vel_x = jnp.where(can_control_puck, 
                               (controlling_player_vel_x * 0.8).astype(jnp.int32),  # 80% of player speed
                               jnp.array(0, dtype=jnp.int32))
        carry_vel_y = jnp.where(can_control_puck,
                               (controlling_player_vel_y * 0.8).astype(jnp.int32),  # 80% of player speed
                               jnp.array(0, dtype=jnp.int32))
        
        # Puck hitting: hit toward goal with horizontal component based on player movement and position
        # Get player position
        player_x = state.players_x[controlling_player_idx]
        player_y = state.players_y[controlling_player_idx]
        
        # Calculate horizontal direction based on player's velocity and position
        # If player is moving horizontally, use that direction
        # Otherwise, try to aim around the center of the goal
        goal_center_x = (self.consts.GOAL_LEFT + self.consts.GOAL_RIGHT) // 2
        dx_to_goal_center = goal_center_x - player_x
        
        hit_dx = jnp.where(
            jnp.abs(controlling_player_vel_x) > 0.5,  # If player is moving horizontally
            jnp.where(controlling_player_vel_x > 0, 1, -1),  # Right if positive, left if negative
            jnp.where(jnp.abs(dx_to_goal_center) > 10,  # If far from goal center
                     jnp.where(dx_to_goal_center > 0, 1, -1),  # Aim toward goal center
                     0)  # No horizontal movement if close to center
        )
        # Vertical direction: always toward goal
        hit_dy = jnp.where(is_team1, 1, -1)  # Team 1 (top) hits down toward enemy goal, Team 2 (bottom) hits up toward player goal
        
        # Calculate hit velocity
        team1_strength = jnp.where(use_stick, 7, 4)  # Player: 7 with stick, 4 without
        hit_strength = jnp.where(is_team1, team1_strength, 6)  # Enemy: always 6
        
        hit_vel_x = jnp.where(can_control_puck, 
                            (hit_dx * hit_strength).astype(jnp.int32),
                            jnp.array(0, dtype=jnp.int32))
        hit_vel_y = jnp.where(can_control_puck,
                            (hit_dy * hit_strength).astype(jnp.int32),
                            jnp.array(0, dtype=jnp.int32))
        
        # Choose between carrying and hitting based on player movement and stick usage
        # If player is using stick, always hit the puck toward goal (ignore movement)
        # If player is moving and not using stick, carry the puck
        # Otherwise, hit the puck
        use_carrying = jnp.logical_and(player_moving, jnp.logical_not(use_stick))
        
        final_vel_x = jnp.where(use_carrying, carry_vel_x, hit_vel_x)
        final_vel_y = jnp.where(use_carrying, carry_vel_y, hit_vel_y)
        
        # Apply velocity to puck (only from controlling player)
        new_puck_vel_x = jnp.where(can_control_puck, final_vel_x, new_puck_vel_x)
        new_puck_vel_y = jnp.where(can_control_puck, final_vel_y, new_puck_vel_y)
        
        # Update puck possession and last touch (only for controlling player)
        players_has_puck = players_has_puck.at[controlling_player_idx].set(jnp.where(can_control_puck, 1, 0))
        new_last_touch = jnp.where(can_control_puck, controlling_player_idx, new_last_touch)
        
        # Limit puck speed and ensure minimum speed (like Pong)
        speed = jnp.sqrt(new_puck_vel_x.astype(jnp.float32)**2 + new_puck_vel_y.astype(jnp.float32)**2)
        
        # Maximum speed limit
        speed_factor = jnp.where(speed > self.consts.PUCK_MAX_SPEED, 
                               self.consts.PUCK_MAX_SPEED / speed, 1.0)
        
        # Minimum speed to keep puck moving (like Pong)
        min_speed = 0.5
        speed_factor = jnp.where(speed < min_speed, 
                               min_speed / jnp.maximum(speed, 0.1), speed_factor)
        
        new_puck_vel_x = (new_puck_vel_x.astype(jnp.float32) * speed_factor).astype(jnp.int32)
        new_puck_vel_y = (new_puck_vel_y.astype(jnp.float32) * speed_factor).astype(jnp.int32)
        
        return new_puck_x, new_puck_y, new_puck_vel_x, new_puck_vel_y, players_has_puck, new_last_touch

    @partial(jax.jit, static_argnums=(0,))
    def _check_goals(self, left_score: chex.Array, right_score: chex.Array, puck_x: chex.Array, puck_y: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Check if a goal was scored and update scores.
        
        Args:
            left_score: Current left team score
            right_score: Current right team score
            puck_x: Puck x position
            puck_y: Puck y position
            
        Returns:
            Tuple of (new_left_score, new_right_score, goal_scored)
        """
        # Check if puck is in goal areas
        in_left_goal = jnp.logical_and(
            jnp.logical_and(puck_x >= self.consts.GOAL_LEFT, puck_x <= self.consts.GOAL_RIGHT),
            puck_y <= self.consts.RINK_TOP + self.consts.GOAL_DEPTH
        )
        
        in_right_goal = jnp.logical_and(
            jnp.logical_and(puck_x >= self.consts.GOAL_LEFT, puck_x <= self.consts.GOAL_RIGHT),
            puck_y >= self.consts.RINK_BOTTOM - self.consts.GOAL_DEPTH
        )
        
        # Update scores (corrected for new positioning)
        # Team 1 (top/player) scores when puck goes into bottom goal (enemy's goal)
        # Team 2 (bottom/enemy) scores when puck goes into top goal (player's goal)
        new_left_score = jnp.where(in_right_goal, left_score + 1, left_score)   # Team 1 (player) scores in bottom goal
        new_right_score = jnp.where(in_left_goal, right_score + 1, right_score)  # Team 2 (enemy) scores in top goal
        
        goal_scored = jnp.logical_or(in_left_goal, in_right_goal)
        
        return new_left_score, new_right_score, goal_scored


class IceHockeyRenderer(JAXGameRenderer):
    #add stick later
    
    def __init__(self, consts: IceHockeyConstants = None):
        """
        Initialize the renderer.
        
        Args:
            consts: Game constants for rendering
        """
        super().__init__()
        self.consts = consts or IceHockeyConstants()
        
        # Load Pong sprites temporarily for testing
        self._load_sprites()
        print("IceHockeyRenderer initialized with Pong sprites (temporary)")

    def _load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        try:
            player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/icehockey/player_normal_r.npy"))
            enemy = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/icehockey/enemy_normal_l.npy"))
            ball = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/icehockey/puck.npy"))
            bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/icehockey/background.npy"))
            logo = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/icehockey/logo.npy"))
            colon = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/icehockey/time.npy"))
            
            player_score_sprites = jr.load_and_pad_digits(
                os.path.join(MODULE_DIR, "sprites/icehockey/player_score_{}.npy"),
                num_chars=10,
            )
            enemy_score_sprites = jr.load_and_pad_digits(
                os.path.join(MODULE_DIR, "sprites/icehockey/enemy_score_{}.npy"),
                num_chars=10,
            )
            
            self.SPRITE_BG = jnp.expand_dims(bg, axis=0)
            self.SPRITE_PLAYER = jnp.expand_dims(player, axis=0)  # Team 1 players
            self.SPRITE_ENEMY = jnp.expand_dims(enemy, axis=0)    # Team 2 players
            self.SPRITE_BALL = jnp.expand_dims(ball, axis=0)      # Puck
            self.SPRITE_LOGO = jnp.expand_dims(logo, axis=0)      # Logo sprite
            self.SPRITE_COLON = jnp.expand_dims(colon, axis=0)    # Colon sprite for timer
            self.PLAYER_DIGIT_SPRITES = player_score_sprites      # Team 1 scores
            self.ENEMY_DIGIT_SPRITES = enemy_score_sprites        # Team 2 scores
            
            print("Pong sprites loaded successfully for ice hockey")
            
        except Exception as e:
            print(f"Warning: Could not load sprites: {e}")
            self._create_fallback_sprites()

    

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: IceHockeyState) -> jnp.ndarray:
        """
        Render the current game state using Pong sprites.
        
        Args:
            state: Current game state
            
        Returns:
            RGB image array
        """
        # Start with background
        raster = jr.create_initial_frame(width=self.consts.WIDTH, height=self.consts.HEIGHT)
        
        # Draw background sprite
        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 40, 40, frame_bg)

        for i in range(2):
            x = state.players_x[i]
            y = state.players_y[i]
            frame_player = jr.get_sprite_frame(self.SPRITE_PLAYER, 0)
            raster = jr.render_at(raster, x - self.consts.PLAYER_WIDTH//2, y - self.consts.PLAYER_HEIGHT//2, frame_player)
        
        for i in range(2, 4):
            x = state.players_x[i]
            y = state.players_y[i]
            frame_enemy = jr.get_sprite_frame(self.SPRITE_ENEMY, 0)
            raster = jr.render_at(raster, x - self.consts.PLAYER_WIDTH//2, y - self.consts.PLAYER_HEIGHT//2, frame_enemy)
        
        puck_x = state.puck_x
        puck_y = state.puck_y
        frame_ball = jr.get_sprite_frame(self.SPRITE_BALL, 0)
        raster = jr.render_at(raster, puck_x - self.consts.PUCK_SIZE//2, puck_y - self.consts.PUCK_SIZE//2, frame_ball)
        
        left_score_digits = jr.int_to_digits(state.left_score, max_digits=2)
        is_left_single_digit = state.left_score < 10
        left_start_index = jax.lax.select(is_left_single_digit, 1, 0)
        left_num_to_render = jax.lax.select(is_left_single_digit, 1, 2)
        left_render_x = jax.lax.select(is_left_single_digit,
                                      self.consts.SCORE_LEFT_X + 8,
                                      self.consts.SCORE_LEFT_X)
        
        raster = jr.render_label_selective(raster, left_render_x, self.consts.SCORE_Y,
                                          left_score_digits, self.PLAYER_DIGIT_SPRITES,
                                          left_start_index, left_num_to_render,
                                          spacing=8)
        
        right_score_digits = jr.int_to_digits(state.right_score, max_digits=2)
        is_right_single_digit = state.right_score < 10
        right_start_index = jax.lax.select(is_right_single_digit, 1, 0)
        right_num_to_render = jax.lax.select(is_right_single_digit, 1, 2)
        right_render_x = jax.lax.select(is_right_single_digit,
                                       self.consts.SCORE_RIGHT_X + 8,
                                       self.consts.SCORE_RIGHT_X)
        
        raster = jr.render_label_selective(raster, right_render_x, self.consts.SCORE_Y,
                                          right_score_digits, self.ENEMY_DIGIT_SPRITES,
                                          right_start_index, right_num_to_render,
                                          spacing=8)
        
        time_remaining = state.time_remaining
        minutes = time_remaining // 60
        seconds = time_remaining % 60
        
        minute_digits = jr.int_to_digits(minutes, max_digits=1)
        second_digits = jr.int_to_digits(seconds, max_digits=2)
        
        minute_render_x = self.consts.TIMER_X
        raster = jr.render_label_selective(raster, minute_render_x, self.consts.TIMER_Y,
                                          minute_digits, self.PLAYER_DIGIT_SPRITES,
                                          0, 1, spacing=8)
        
        colon_x = minute_render_x + 8
        frame_colon = jr.get_sprite_frame(self.SPRITE_COLON, 0)
        raster = jr.render_at(raster, colon_x, self.consts.TIMER_Y, frame_colon)
        
        seconds_render_x = colon_x + 8
        raster = jr.render_label_selective(raster, seconds_render_x, self.consts.TIMER_Y,
                                          second_digits, self.PLAYER_DIGIT_SPRITES,
                                          0, 2, spacing=8)
        
        frame_logo = jr.get_sprite_frame(self.SPRITE_LOGO, 0)
        raster = jr.render_at(raster, self.consts.LOGO_X, self.consts.LOGO_Y, frame_logo)
        
        return raster


