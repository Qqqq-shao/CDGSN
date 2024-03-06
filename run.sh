python CDGSN_main.py --num-heads=8 --num-layers=7 --num-hidden=64 --in-drop=0.6 --attn-drop=0.4 --Weight-node-clf=1 --l2-w=0.0005 --lr-ini=0.001 --Weight-ada=0.1 --Weight-edge-clf=1 --source='acmv9'  --target='citationv1' --edge-type='concat' --Weight-cda=0.1 --Weight-SelPNL=0.1 --threshold_pos=0.8 --threshold_neg=0.2

python CDGSN_main.py --num-heads=8 --num-layers=6 --num-hidden=64 --in-drop=0.6 --attn-drop=0.0 --Weight-node-clf=1 --l2-w=0.0005 --lr-ini=0.001 --Weight-ada=0.1 --Weight-edge-clf=1 --source='acmv9'  --target='dblpv7' --edge-type='concat' --Weight-cda=0.1 --Weight-SelPNL=0.01 --threshold_pos=0.8 --threshold_neg=0.2

python CDGSN_main.py --num-heads=8 --num-layers=7 --num-hidden=64 --in-drop=0.6 --attn-drop=0.3 --Weight-node-clf=1 --l2-w=0.0001 --lr-ini=0.001 --Weight-ada=0.1 --Weight-edge-clf=1 --source='citationv1'  --target='acmv9' --edge-type='concat' --Weight-cda=0.1 --Weight-SelPNL=0.1 --threshold_pos=0.8 --threshold_neg=0.2

python CDGSN_main.py --num-heads=8 --num-layers=7 --num-hidden=64 --in-drop=0.6 --attn-drop=0.1 --Weight-node-clf=1 --l2-w=0.0005 --lr-ini=0.001 --Weight-ada=0.1 --Weight-edge-clf=1 --source='citationv1'  --target='dblpv7' --edge-type='concat' --Weight-cda=1 --Weight-SelPNL=0.01 --threshold_pos=0.8 --threshold_neg=0.2

python CDGSN_main.py --num-heads=8 --num-layers=3 --num-hidden=64 --in-drop=0.6 --attn-drop=0.4 --Weight-node-clf=0.01 --l2-w=0.001 --lr-ini=0.001 --Weight-ada=0.1 --Weight-edge-clf=1 --source='dblpv7'  --target='acmv9' --edge-type='concat' --Weight-cda=1 --Weight-SelPNL=0.001 --threshold_pos=0.8 --threshold_neg=0.2

python CDGSN_main.py --num-heads=8 --num-layers=6 --num-hidden=64 --in-drop=0.6 --attn-drop=0.0 --Weight-node-clf=1 --l2-w=0.0005 --lr-ini=0.001 --Weight-ada=0.1 --Weight-edge-clf=1 --source='dblpv7'  --target='citationv1' --edge-type='concat' --Weight-cda=1 --Weight-SelPNL=0.1 --threshold_pos=0.8 --threshold_neg=0.2
