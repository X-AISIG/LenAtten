while getopts "tvrT" switches; do
    case "$switches" in
        t)
            echo 'Perform Training MLE'
            python -W ignore train.py --train_mle=yes --train_rl=no  --mle_weight=1 --prefix LA250_256
            # choice: 
            #   --prefix: storage file prefix.
            ;;
        v)
            echo 'Perform Cross validation' 
			python -W ignore eval.py --task=validate --start_from 0010000.tar --prefix LA250_256
            # choice:
            # --len_attn_visual: display length attention hotplot.
            ;;
        r)
            echo 'Perform Training RL' 
			python -W ignore train.py --train_mle=yes --train_rl=yes  --mle_weight=0.25 --load_model classfier_00/0050000.tar --new_lr 0.0001 --prefix classfier_00_rl_nolenreward
            ;;
        T)
            echo 'Perform Test' 
            # python -W ignore eval.py --task=test --start_from 0080000.tar --prefix classifier_01_RL_nolenR
            # python -W ignore eval.py --task=test --start_from 0050000.tar --prefix LA200_256
            python -W ignore eval.py --task=test --start_from 0050000.tar --prefix LA250_256
			;;
        \?)
            echo -e "Invalid option: -$OPTARG."
            show_help
            exit 1
            ;;
    esac
done
