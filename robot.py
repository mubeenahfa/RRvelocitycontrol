import numpy as np


class PlanarRRRobot:
    def __init__(
        self,
        l1=3.0,
        l2=2.0,
        theta1_limits=(-90, 90),
        theta2_limits=(-170, 170),
        singularity_threshold=1e-4,
    ):
        self.l1 = l1
        self.l2 = l2

        # im converting to radians here
        self.theta1_limits = np.deg2rad(theta1_limits)
        self.theta2_limits = np.deg2rad(theta2_limits)

        self.singularity_threshold = singularity_threshold

    def forward_kinematics(self, theta):
        """
        Im using this function to calculate end effector from joint angles
        """
        theta1, theta2 = theta

        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)

        return np.array([x, y])

    def joint_positions(self, theta):
        """
        Rreturns immportant data that I will use for plotting
        """
        theta1, theta2 = theta

        base = np.array([0.0, 0.0])

        elbow = np.array([
            self.l1 * np.cos(theta1),
            self.l1 * np.sin(theta1),
        ])

        end_effector = elbow + np.array([
            self.l2 * np.cos(theta1 + theta2),
            self.l2 * np.sin(theta1 + theta2),
        ])

        return base, elbow, end_effector

    def jacobian(self, theta):
        """
        returns my jacobian matrix. I have shown derivation on latex document
        """
        theta1, theta2 = theta

        j11 = -self.l1 * np.sin(theta1) - self.l2 * np.sin(theta1 + theta2)
        j12 = -self.l2 * np.sin(theta1 + theta2)

        j21 = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        j22 = self.l2 * np.cos(theta1 + theta2)

        return np.array([
            [j11, j12],
            [j21, j22],
        ])

    def determinant(self, theta):
        """
        this helper function returns determinant so i can check singularity
        """
        J = self.jacobian(theta)
        return np.linalg.det(J)

    def is_singular(self, theta):
        """
        raises alarm if robot is close to singularity
        """
        det_j = self.determinant(theta)
        return abs(det_j) < self.singularity_threshold

    def check_joint_limits(self, theta):
        """
        to check if joints are inside the limits
        """
        theta1, theta2 = theta

        theta1_ok = self.theta1_limits[0] <= theta1 <= self.theta1_limits[1]
        theta2_ok = self.theta2_limits[0] <= theta2 <= self.theta2_limits[1]

        return theta1_ok and theta2_ok

    def clamp_to_joint_limits(self, theta):
        """
        just a clipping function that I have added
        """
        theta1, theta2 = theta

        theta1 = np.clip(theta1, self.theta1_limits[0], self.theta1_limits[1])
        theta2 = np.clip(theta2, self.theta2_limits[0], self.theta2_limits[1])

        return np.array([theta1, theta2])