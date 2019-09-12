 open terminal: 1)roscore & 2)rosrun gazebo_ros gazeborun ros 3)rosrun rviz rviz
run main.py
set following parameter in codes
objectNum = 2 #set 1 single object situation set 2 multiple object
    maxIteration = 30 #registration iteration time
    tolerance = 0.0005 #error tolerance for registration
    controlPoints = 500 #maximum sample point number for ICP
    turtlename = 'beer' #tagrget name to be tracked
    Imgtemp='beer.jpg' # recognition template
    template='beer.pcd' #ICP tmeplate
