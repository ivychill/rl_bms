cd ~/project/digit_rec
python digit_main.py&
cd ~/project/image_rec
python ai_main.py&
cd ~/project/fly_control_bms
python AiMain.py&
sleep 1
cd ~/project/ai_bms
python a3c.py&