from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Ovo je GStreamer pipeline koji prima video stream sa BlueROV2
    # Sluša na UDP portu 5600, dekodira H.264 video i prosleđuje ga ROS-u
    gscam_pipeline = (
       'udpsrc port=5600 ! application/x-rtp, media=video, clock-rate=90000, '
       'encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! '
        
       # Namjestavanje FPS 
       'videorate ! video/x-raw,framerate=15/1 ! '
        
       # Namjestavanje rezolucije
       'videoscale ! video/x-raw,width=1920,height=1080 ! '
        
       # Prebacivanje boje u format za ROS
       'videoconvert'
    )

    # gscam_pipeline = (
    #    'v4l2src device=/dev/video0 ! videoconvert'
    # )

    # gscam node
    gscam_node = Node(
        package='gscam',
        executable='gscam_node',
        name='camera',
        parameters=[
            {'gscam_config': gscam_pipeline},
            {'use_gst_timestamps': True},
            {'image_encoding': 'rgb8'},
            {'preroll': True},
        ],
        output='screen'
    )

    return LaunchDescription([
        gscam_node,
    ])
