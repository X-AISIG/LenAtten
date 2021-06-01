while getopts "tvrT" switches; do
    case "$switches" in
        t)
            echo 'Perform Training MLE'
            python -W ignore train.py --train_mle=yes --train_rl=no  --mle_weight=1 --prefix baseline_256
            # choice: 
            #   --prefix: storage file prefix.
            ;;
        v)
            echo 'Perform Cross validation' 
			python -W ignore eval.py --task=validate --start_from 0010000.tar --prefix baseline_256
            ;;
        r)
            echo 'Perform Training RL' 
			python -W ignore train.py --train_mle=yes --train_rl=yes  --mle_weight=0.25 --load_model 1026_1700/0010000.tar --new_lr 0.0001 --prefix 1027_RL_no+lenreward
            ;;
        T)
            echo 'Perform Test' 
			python -W ignore eval.py --task=test --start_from 0050000.tar --prefix baseline_256
            ;;
        \?)
            echo -e "Invalid option: -$OPTARG."
            show_help
            exit 1
            ;;
    esac
done
