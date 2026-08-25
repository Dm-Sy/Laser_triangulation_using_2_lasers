import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaserPerceptionNodeSimulation(Node):
    def __init__(self):
        super().__init__('laser_perception_node_simulation')

        # --- PARAMETRI ZA CRVENU BOJU (HSV) ---
        # Crvena boja prelazi preko granice 0 u HSV-u, pa trebamo dva raspona
        self.declare_parameter('h_min_1', 0)
        self.declare_parameter('s_min_1', 85)
        self.declare_parameter('v_min_1', 100)
        self.declare_parameter('h_max_1', 40)
        self.declare_parameter('s_max_1', 255)
        self.declare_parameter('v_max_1', 255)

        self.declare_parameter('h_min_2', 140)
        self.declare_parameter('s_min_2', 60)
        self.declare_parameter('v_min_2', 100)
        self.declare_parameter('h_max_2', 180)
        self.declare_parameter('s_max_2', 255)
        self.declare_parameter('v_max_2', 255)
        # Minimalna i maksimalna površina točke (pikseli)
        self.declare_parameter('min_area', 0.3)
        self.declare_parameter('max_area', 10000.0)
        # Minimalni razmak po X i maksimalni razmak po Y (pikseli)
        self.declare_parameter('max_y_diff', 150)
        self.declare_parameter('min_x_diff', 10)

        self.bridge = CvBridge()

        self.cx = 960  # Optički centar očitan iz matrice kamere
        
        # --- PUBLISHERI & PRETPLATE ---
        self.points_pub = self.create_publisher(Float64MultiArray, '/laser_points', 10)
        self.debug_pub = self.create_publisher(Image, '/laser_perception/debug_image', 10)
        self.mask_pub = self.create_publisher(Image, '/laser_perception/debug_mask', 10)
        
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        self.get_logger().info('Laser Perception Node pokrenut. Tražim crvene konture...')

    def find_laser_candidates(self, hsv_image):
        # Dohvaćanje parametara za prvi raspon crvene
        lower_red_1 = np.array([self.get_parameter('h_min_1').value, self.get_parameter('s_min_1').value, self.get_parameter('v_min_1').value])
        upper_red_1 = np.array([self.get_parameter('h_max_1').value, self.get_parameter('s_max_1').value, self.get_parameter('v_max_1').value])
        # Dohvaćanje parametara za drugi raspon crvene
        lower_red_2 = np.array([self.get_parameter('h_min_2').value, self.get_parameter('s_min_2').value, self.get_parameter('v_min_2').value])
        upper_red_2 = np.array([self.get_parameter('h_max_2').value, self.get_parameter('s_max_2').value, self.get_parameter('v_max_2').value])
        
        mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
        
        # Spajanje maski
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Čišćenje šuma
        kernel = np.ones((5, 5), np.uint8) # Sve što je manje/jednako od 5x5 piksela bit će očišćeno
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Izdvajanje kanala svjetline (Value kanal iz HSV-a) za podpikselnu analizu
        v_channel = hsv_image[:, :, 2]
        
        candidates = []
        min_area = self.get_parameter('min_area').value
        max_area = self.get_parameter('max_area').value
        # Vračanje kandidata koji zadovoljavaju kriterije
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area: 
                # Pronalaženje okvira (Bounding Box) oko konture radi bržeg računanja
                x, y, w, h = cv2.boundingRect(c)
                
                # Izdvajanje dijela slike sa svjetlinom (V kanal) samo za taj okvir
                roi_v = v_channel[y:y+h, x:x+w]
                
                # Stvaranje lokalnu masku samo za tu konturu kako se ne bi zahvatio okolni šum
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                shifted_contour = c - [x, y]
                cv2.drawContours(roi_mask, [shifted_contour], -1, 255, -1)
                
                # Primijena maske na V kanal kako bi se dobile težine samo za laser
                roi_weighted = cv2.bitwise_and(roi_v, roi_v, mask=roi_mask)
                
                # Izračun momenata (težište) na temelju intenziteta slike
                M = cv2.moments(roi_weighted)
                
                if M["m00"] > 0:
                    # Podpikselne koordinate (decimale) - vraćamo offset 'x' i 'y' iz okvira
                    cx = (M["m10"] / M["m00"]) + x
                    cy = (M["m01"] / M["m00"]) + y
                    
                    # Candidates sadrži (float, float, float, array)
                    candidates.append((cx, cy, area, c))
                    
        return candidates, mask
    
    def select_best_pair(self, candidates):
        if len(candidates) < 2: return None
        
        best_pair = None
        best_score = float('inf')
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                p1, p2 = candidates[i], candidates[j]
                
                y_diff = abs(p1[1] - p2[1])
                x_diff = abs(p1[0] - p2[0])

                max_y = self.get_parameter('max_y_diff').value
                min_x = self.get_parameter('min_x_diff').value
                
                # Budući da su laseri na otprilike istom vodoravnom pravcu, y_diff mora biti relativno mali
                if y_diff < max_y and x_diff > min_x:
                    score = y_diff 
                    area1, area2 = p1[2], p2[2]
                    area_ratio = min(area1, area2) / max(area1, area2)
                    area_penalty = (1.0 - area_ratio) * 100 
                    
                    score += area_penalty
                    
                    if score < best_score:
                        best_score = score
                        # Sortira tako da je prvi uvijek lijevi laser, drugi desni
                        if p1[0] < p2[0]:
                            best_pair = (p1, p2)
                        else:
                            best_pair = (p2, p1)
                        
        return best_pair
    
    def image_callback(self, msg):
        try:
            # Kako se prima 'Image', umjesto 'CompressedImage', koristi se CvBridge za konverziju
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
        except Exception as e:
            self.get_logger().error(f"Greška pri dekodiranju slike: {e}")
            return

        debug_image = cv_image.copy()
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv, 5)

        candidates, mask = self.find_laser_candidates(hsv)
        pair = self.select_best_pair(candidates)

        if pair:
            p_left, p_right = pair

            # Prikazivanje optičkog centra
            height = debug_image.shape[0]
            # Može se dodati dinamički cx ili koristi self.cx
            trenutni_cx = int(debug_image.shape[1] / 2)
            # trenutni_cx = int(self.cx)
            cv2.line(debug_image, (trenutni_cx, 0), (trenutni_cx, height), (0, 0, 255), 2)
            cv2.putText(debug_image, "OS KAMERE", (trenutni_cx + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Pretvaranje decimalnih koordinata u cijele brojeve za crtanje
            cx_left_int, cy_left_int = int(round(p_left[0])), int(round(p_left[1]))
            cx_right_int, cy_right_int = int(round(p_right[0])), int(round(p_right[1]))
            
            # Crtanje kontura i središta za debug
            cv2.drawContours(debug_image, [p_left[3]], -1, (0, 255, 0), 2)
            cv2.drawContours(debug_image, [p_right[3]], -1, (0, 255, 0), 2)
            
            # Korištenje int vrijednosti za crtanje na slici
            cv2.circle(debug_image, (cx_left_int, cy_left_int), 4, (255, 0, 0), -1)
            cv2.circle(debug_image, (cx_right_int, cy_right_int), 4, (255, 0, 0), -1)
            cv2.putText(debug_image, "L", (cx_left_int-20, cy_left_int-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(debug_image, "R", (cx_right_int-20, cy_right_int-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Slanje float koordinata u kinematiku
            points_msg = Float64MultiArray()
            points_msg.data = [float(p_left[0]), float(p_left[1]), float(p_right[0]), float(p_right[1])]
            self.points_pub.publish(points_msg)

        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = LaserPerceptionNodeSimulation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
