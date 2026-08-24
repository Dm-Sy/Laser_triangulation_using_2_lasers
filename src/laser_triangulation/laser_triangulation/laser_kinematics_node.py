import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
import numpy as np
import cv2
import math

class LaserKinematicsNode(Node):
    def __init__(self):
        super().__init__('laser_kinematics_node')

        # --- KONFIGURACIJA KAMERE I LASERA ---
        # Udaljenost između lasera
        self.B_METERS = 0.205  
        # Kalibrirane fizičke udaljenosti (u metrima) lasera od optičkog centra kamere
        self.declare_parameter('x_left', -0.1462)
        self.declare_parameter('x_right', 0.0588)
        # Matrica kamere
        self.CAM_MATRIX = np.array([
            [1101.064731353069, 0.000000, 903.5475221887251],
            [0.000000, 1102.243517934071, 526.5995712488191],
            [0.000000, 0.000000, 1.000000]
        ], dtype=np.float32)

        self.DIST_COEFFS = np.array([0.0561898897739297, -0.017629664177420498, -0.001807160045386734, -0.0018453430876528153, 0.000000], dtype=np.float32)

        # Izdvajanje fokalne duljine i optičkog centra
        self.fx = self.CAM_MATRIX[0, 0]
        self.cx = self.CAM_MATRIX[0, 2]

        # --- PUBLISHERI I PRETPALTE ---
        # Stvarna, najkraća okomita udaljenost do zida
        self.perp_dist_pub = self.create_publisher(Float64, '/robot/wall_distance_perpendicular', 10)
        # Udaljenost do točke promatranja (po osi kamere)
        self.obs_dist_pub = self.create_publisher(Float64, '/robot/wall_distance_observed', 10)
        # Kut zakreta
        self.yaw_pub = self.create_publisher(Float64, '/robot/wall_yaw_angle', 10)
        
        self.points_sub = self.create_subscription(Float64MultiArray, '/laser_points', self.points_callback, 10)
        
        self.get_logger().info('Laser Kinematics Node pokrenut.')

    def points_callback(self, msg):
        if len(msg.data) < 4:
            return

        # Dohvaćanje točaka: (u1, v1) je lijeva, (u2, v2) je desna
        pts_distorted = np.array([
            [[msg.data[0], msg.data[1]]],
            [[msg.data[2], msg.data[3]]]
        ], dtype=np.float32)

        # Uklanjanje distorzije
        pts_undistorted = cv2.undistortPoints(pts_distorted, self.CAM_MATRIX, self.DIST_COEFFS, P=self.CAM_MATRIX)
        
        u_left = pts_undistorted[0, 0, 0]
        u_right = pts_undistorted[1, 0, 0]

        # Provjera kako bi se izbjeglo dijeljenje s nulom (ako je točka točno u centru)
        if (u_left - self.cx) == 0 or (u_right - self.cx) == 0:
            return

        # Računanje pojedinačnih dubina za svaku lasersku zraku
        # Pretpostavka da su laseri simetrično razmaknuti za B/2 u odnosu na objektiv kamere
        # X_left = -self.B_METERS / 2.0
        # X_right = self.B_METERS / 2.0
        
        X_left = self.get_parameter('x_left').value
        X_right = self.get_parameter('x_right').value

        Z_left = (self.fx * X_left) / (u_left - self.cx)
        Z_right = (self.fx * X_right) / (u_right - self.cx)

        # Odbacivanje nemogućih (negativnih) vrijednosti uslijed šuma
        if Z_left <= 0 or Z_right <= 0:
            return

        # Računanje kuta (yaw)
        # Ako je lijevi Z > desni Z, to znači da je robot je okrenut ulijevo
        dz = Z_right - Z_left
        yaw_rad = math.atan2(dz, self.B_METERS)
        yaw_deg = math.degrees(yaw_rad)

        # Udaljenost do točke promatranja (po osi kamere)
        Z_observed = (Z_left + Z_right) / 2.0

        # Stvarna, najkraća okomita udaljenost do ravnine zida
        Z_perpendicular = Z_observed * math.cos(yaw_rad)

        # Objavljivanje podataka
        self.obs_dist_pub.publish(Float64(data=Z_observed))
        self.perp_dist_pub.publish(Float64(data=Z_perpendicular))
        self.yaw_pub.publish(Float64(data=yaw_deg))

        # Ispis dobivenih vrijednosti u terminal
        self.get_logger().info(f"Obs Dist: {Z_observed:.3f}m | Perp Dist: {Z_perpendicular:.3f}m | Yaw: {yaw_deg:.2f}°")

def main(args=None):
    rclpy.init(args=args)
    node = LaserKinematicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
