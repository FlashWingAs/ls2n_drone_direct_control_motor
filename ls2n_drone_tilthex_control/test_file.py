from tilthex_common import *
from tilthex_controllers import *
from tilthex_control import *

# TEST 1 : Translationnal acceleration check

print("\n\nTEST 1\n\n")
rclpy.init(args=None)
node = TilthexControlCenter()
test_controller = Geometric_Controller(node=node)

Acc = np.array([10, 0, 10])
Pose = Tilthex_Pose()
Pose.rotation = Quaternion([0.9659258, 0.258819, 0, 0])
test_controller.translationnal_acceleration_check(Acc, Pose)

# TEST 2 : Observer check
print("\n\nTEST 2\n\n")
I = np.diag([1, 1, 1.5])
observer = Tilthex_Observer(3.981, I)