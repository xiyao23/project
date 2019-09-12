Step1 Open terminal Launch Gazebo and Ros:
roscore &
rosrun gazebo_ros gazebo
rosrun rviz rviz
Step2 Simulate objects in the Gazebo
Step3 Set Pointcloud2 and TF in Rviz
Step4 Set parameters in codes:

set following parameter in codes
objectNum = 2 #set 1 single object situation set 2 multiple object
    maxIteration = 30 #registration iteration time
    tolerance = 0.0005 #error tolerance for registration
    controlPoints = 500 #maximum sample point number for ICP
    turtlename = 'beer' #tagrget name to be tracked
    Imgtemp='beer.jpg' # recognition template
    template='beer.pcd' #ICP tmeplate
run main.py
