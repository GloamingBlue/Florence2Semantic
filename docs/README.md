# Florence2Semantic
This project is used for video semantic generation of the RealSense camera on the mechanical guide dog.  
[Document](https://www.yuque.com/u57806034/mdfxam/cobhfh7nmiphlgbg)  
[Usage](https://github.com/GloamingBlue/Florence2Semantic/blob/main/docs/ROS2_USAGE.md)  
florence2_caption.py是语义分析功能的最小实现，适用于分析本地图像进行测试；  
florence2_caption_ros2.py将前者集成到ros2中，实时监测视频流接收控制信号进行语义分析，并且增加了翻译功能，支持性能监测；  
florence2_caption_ros2_lite.py将前者的性能监测功能去掉，只保留项目需要的部分；  
configs/florence2_caption_params.yaml是前两者的配置文件，并且兼容命令行参数配置。  
