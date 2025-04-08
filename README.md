# medical_PRM

1. Training Dataset Download
huggingface에서 jhyun0414/train_dataset 를 다운로드해서
./dataset/2_train_dataset.json 로 저장


2. PRM finetuning 
instruct model: ./shell/5_0401_PRM_finetuning.sh 실행
base model: ./shell/6_0408_PRM_finetuning_vanilla.sh 실행

세부 설정은 sh 파일에서 조절 가능능