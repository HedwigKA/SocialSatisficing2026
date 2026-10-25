import random
import unittest

import numpy as np

import env
import model
import model_Qshare
import model_RS


class TestSuboptimaWorld(unittest.TestCase):
    def test_default_reward_placement(self):
        world = env.SuboptimaWorld(row=9, col=9, start=40)
        expected = {
            (0, 2): 8,
            (0, 6): 1,
            (2, 0): 3,
            (2, 8): 5,
            (6, 0): 6,
            (6, 8): 4,
            (8, 2): 2,
            (8, 6): 7,
        }
        for coord, reward in expected.items():
            self.assertEqual(world.world[coord], reward)

    def test_shuffled_reward_assignment(self):
        shuffled = [1, 2, 3, 4, 5, 6, 7, 8]
        world = env.SuboptimaWorld(row=9, col=9, start=40, reward_values=shuffled)
        placed_rewards = [world.world[r, c] for (r, c) in world.goal_coordinates]
        self.assertEqual(placed_rewards, shuffled)
        self.assertEqual(set(placed_rewards), set(range(1, 9)))

    def test_boundary_is_noop(self):
        world = env.SuboptimaWorld(row=9, col=9, start=40)
        top_left = world.coord_to_state(0, 0)
        next_state, _, _ = world.step(top_left, 0)  # up at top boundary
        self.assertEqual(next_state, top_left)
        next_state, _, _ = world.step(top_left, 2)  # left at left boundary
        self.assertEqual(next_state, top_left)


class TestQLearningUpdates(unittest.TestCase):
    def test_non_terminal_update_bootstraps(self):
        q = model.Q_learning(learning_rate=0.1, discount_rate=0.9, num_state=2, num_action=2)
        q.Q[1] = np.array([2.0, 1.0])
        q.update_Q(current_state=0, next_state=1, current_action=0, reward=1.0, done=False)
        # target: 1 + 0.9 * 2 = 2.8, update: 0 + 0.1 * 2.8
        self.assertAlmostEqual(q.Q[0, 0], 0.28)

    def test_terminal_update_no_bootstrap(self):
        q = model.Q_learning(learning_rate=0.1, discount_rate=0.9, num_state=2, num_action=2)
        q.Q[1] = np.array([10.0, 10.0])
        q.update_Q(current_state=0, next_state=1, current_action=0, reward=1.0, done=True)
        # terminal target should be reward only
        self.assertAlmostEqual(q.Q[0, 0], 0.1)

    def test_terminal_update_qshare_model(self):
        q = model_Qshare.Q_learning(learning_rate=0.1, discount_rate=0.9, num_state=2, num_action=2)
        q.Q[1] = np.array([10.0, 10.0])
        q.update_Q(current_state=0, next_state=1, current_action=0, reward=1.0, done=True)
        self.assertAlmostEqual(q.Q[0, 0], 0.1)


class TestRSAndGRC(unittest.TestCase):
    def test_rs_tie_break_action_in_tied_set(self):
        random.seed(0)
        rs = model.RS(
            learning_rate=0.1,
            discount_rate=0.9,
            reference=0.0,
            tau_alpha=0.1,
            tau_gamma=0.9,
            num_state=1,
            num_action=4,
        )
        rs.value_func.Q[0] = np.array([1.0, 1.0, 0.0, -1.0])
        rs.tau[0] = np.array([1.0, 1.0, 1.0, 1.0])
        action = rs.get_select_action(0)
        self.assertIn(action, [0, 1])

    def test_grc_reference_conversion_consistency(self):
        random.seed(1)
        grc = model.GRC(
            learning_rate=0.1,
            discount_rate=0.9,
            reference=6.5,
            tau_alpha=0.1,
            tau_gamma=0.9,
            EG_gamma=0.9,
            num_state=1,
            num_action=2,
        )
        grc.value_func.Q[0] = np.array([1.0, 2.0])
        grc.tau[0] = np.array([1.0, 1.0])
        grc.EG = 4.0
        grc.RG = 6.5
        expected_reference = np.max(grc.value_func.Q[0]) - min(grc.EG - grc.RG, 0)
        rs_values = grc.tau[0] * (grc.value_func.Q[0] - expected_reference)
        action = grc.get_select_action(0)
        self.assertIn(action, np.where(rs_values == np.max(rs_values))[0])


class TestSharing(unittest.TestCase):
    def test_q_sharing_takes_elementwise_max(self):
        a0 = model_Qshare.agent(0.1, 0.9, 600, num_state=2, num_action=2)
        a1 = model_Qshare.agent(0.1, 0.9, 600, num_state=2, num_action=2)
        a0.value_func.Q = np.array([[1.0, 4.0], [0.5, 2.0]])
        a1.value_func.Q = np.array([[2.0, 3.0], [1.0, 1.5]])
        a0.share_Q([a0, a1])
        expected = np.array([[2.0, 4.0], [1.0, 2.0]])
        np.testing.assert_array_equal(a0.value_func.Q, expected)


if __name__ == "__main__":
    unittest.main()
