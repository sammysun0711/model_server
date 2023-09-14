echo "1. Install VLC for RTSP Server.."
sudo apt-get install vlc

echo "2. Init python environment ..."
python3 -m venv devcon_demo 
source devcon_demo/bin/activate

echo "3. Install python dependency ..."
pip3 install -r requirements.txt 

echo "4. Download vehicle detection model from OpenVINO Open Model Zoo ..."
mkdir -p models/1
wget -P models/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/vehicle-detection-0202/FP16-INT8/vehicle-detection-0202.xml
wget -P models/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/vehicle-detection-0202/FP16-INT8/vehicle-detection-0202.bin
