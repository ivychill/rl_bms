cd ~/project/image_rec
python main.py&
cd ~/project/fly_control_bms
python MainFunc.py&
sleep 1
cd ~/project/ai_bms
python a3c.py&
