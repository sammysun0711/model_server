echo "1. Install VLC for RTSP Server.."
sudo apt-get install vlc
echo " "

echo "2. Init python virtual environment and install python dependency ..."
python3 -m venv devcon_demo 
source devcon_demo/bin/activate
pip3 install -r requirements.txt 
echo " "

echo "3. Download vehicle detection model from OpenVINO Open Model Zoo ..."
mkdir -p models/1
wget -P models/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/vehicle-detection-0202/FP16-INT8/vehicle-detection-0202.xml
wget -P models/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/vehicle-detection-0202/FP16-INT8/vehicle-detection-0202.bin
