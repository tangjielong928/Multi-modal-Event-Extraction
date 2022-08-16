dataset_name=EE1.0
data_dir=./data/${dataset_name}
#conf_dir=./conf/${dataset_name}
conf_dir=./data/${dataset_name}
ckpt_dir=./ckpt/${dataset_name}
#submit_data_path=./submit/test_duee_1.json

learning_rate=5e-5
max_seq_len=300
batch_size=32
epoch=60

echo -e "check and create directory"
dir_list=(./ckpt ${ckpt_dir} ./submit)
for item in ${dir_list[*]}
do
    if [ ! -d ${item} ]; then
        mkdir ${item}
        echo "create dir * ${item} *"
    else
        echo "dir ${item} exist"
    fi
done

process_name=${1}
pred_data=${data_dir}/multiLabel_result.json   # 多标签预测结果

run_role_labeling_model(){
    model=${1}
    is_train=${2}
    pred_save_path=../data/result/${model}
    cd Event_extraction || { echo "Enter Failure"; exit 1; }
    sh run_role_labeling.sh .${data_dir}/${model} .${conf_dir}/${model}_tag.dict .${ckpt_dir}/${model} .${pred_data} ${learning_rate} ${is_train} ${max_seq_len} ${batch_size} ${epoch} ${pred_save_path}
}

run_trigger_labeling_model(){
    model=${1}
    is_train=${2}
    pred_save_path=../data/result/${model}
    cd Event_extraction || { echo "Enter Failure"; exit 1; }
    sh run_trigger_labeling.sh .${data_dir}/${model} .${conf_dir}/${model}_tag.dict .${ckpt_dir}/${model} .${pred_data} ${learning_rate} ${is_train} ${max_seq_len} ${batch_size} ${epoch} ${pred_save_path}
}

if [ ${process_name} == data_prepare ]; then
    echo -e "\nstart ${dataset_name} data prepare"
    cd Event_extraction || { echo "Enter Failure"; exit 1; }
    python duee_1_data_prepare.py
    echo -e "end ${dataset_name} data prepare"
elif [ ${process_name} == data_augmentation ]; then
    echo -e "\nstart ${dataset_name} data augmentation"
    cd Event_extraction || { echo "Enter Failure"; exit 1; }
    python data_agumentation.py
    echo -e "end ${dataset_name} data augmentation"
elif [ ${process_name} == multi_label_train ]; then
    echo -e "\nstart  multi_label train"
    cd bert_multi_classification || { echo "Enter Failure"; exit 1; }
    python train.py
    echo -e "end  multi_label train"
elif [ ${process_name} == multi_label_predict ]; then
    echo -e "\nstart  multi_label predict"
    cd bert_multi_classification || { echo "Enter Failure"; exit 1; }
    python predict_result.py
    echo -e "end multi_label predict"
elif [ ${process_name} == trigger_train ]; then
    echo -e "\nstart ${dataset_name} trigger train"
    run_trigger_labeling_model trigger True
    echo -e "end ${dataset_name} trigger train"
elif [ ${process_name} == trigger_predict ]; then
    echo -e "\nstart ${dataset_name} trigger predict"
    run_trigger_labeling_model trigger False
    echo -e "end ${dataset_name} trigger predict"
elif [ ${process_name} == role_train ]; then
    echo -e "\nstart ${dataset_name} role train"
    run_role_labeling_model role True
    echo -e "end ${dataset_name} role train"
elif [ ${process_name} == role_predict ]; then
    echo -e "\nstart ${dataset_name} role predict"
    run_role_labeling_model role False
    echo -e "end ${dataset_name} role predict"
elif [ ${process_name} == objDec_train ]; then
    echo -e "\nstart yolov5 train"
    cd yolov5 || { echo "Enter Failure"; exit 1; }
    python train.py
    echo -e "end  yolov5 train"
elif [ ${process_name} == objDec_predict ]; then
    echo -e "\nstart yolov5 predict"
    cd yolov5 || { echo "Enter Failure"; exit 1; }
    python predict.py
    echo -e "end  yolov5 predict"
elif [ ${process_name} == pred_2_submit ]; then
    echo -e "\nstart ${process_name} predict data merge to submit fotmat"
    python postprocess.py
    echo -e "end ${process_name}  predict data merge"
elif [ ${process_name} == pipeline ]; then
    echo -e "\nstart ${process_name} process, it will take some time!!"
    #启动多标签多分类预测
    echo -e "\nstart  multi_label predict"
    cd bert_multi_classification || { echo "Enter Failure"; exit 1; }
    python predict_result.py
    cd ../ || { echo "Enter Failure"; exit 1; }
    echo -e "end multi_label predict"
    #启动触发词预测
    echo -e "\nstart ${dataset_name} trigger predict"
    run_trigger_labeling_model trigger False
    cd ../ || { echo "Enter Failure"; exit 1; }
    echo -e "end ${dataset_name} trigger predict"
    #启动论元预测
    echo -e "\nstart ${dataset_name} role predict"
    run_role_labeling_model role False
    cd ../ || { echo "Enter Failure"; exit 1; }
    echo -e "end ${dataset_name} role predict"
    #启动目标检测预测
    echo -e "\nstart yolov5 predict"
    cd yolov5 || { echo "Enter Failure"; exit 1; }
    python predict.py
    cd ../ || { echo "Enter Failure"; exit 1; }
    echo -e "end  yolov5 predict"
    #启动后处理
    echo -e "\nstart ${process_name} predict data merge to submit fotmat"
    python postprocess.py
    echo -e "end ${process_name}  predict data merge"

    echo -e "end ${process_name} process"
else
    echo "no process name ${process_name}"
fi