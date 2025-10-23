# prepare api key and secret form https://www.faceplusplus.com.cn/
api_key=""
api_secret=""


# generate celeba face aging results for Face++ eval, Age accuracy eval, Arcface eval
python -m lib.eval.infer_dataset \
  --models_dir "models/Cradle2Cane" \
  --input_folder "dataset/eval/celeba-200" \
  --output_dir "outputs/celeba-200" 

# generate AgeDB face aging results for LPIPS eval
python -m lib.eval.infer_dataset \
  --models_dir "models/Cradle2Cane" \
  --input_folder "dataset/eval/agedb-400" \
  --output_dir "outputs/agedb-400" 

# face++_eval
python -m lib.eval.face++_eval \
  --folder1 "dataset/eval/celeba-200" \
  --folder2 "outputs/celeba-200" \
  --api_key ${api_key} \
  --api_secret ${api_secret} \

# acrface_eval
python -m lib.eval.arcface_eval \
  --folder1 "dataset/eval/celeba-200" \
  --folder2 "outputs/celeba-200" \
  --weight models/backbone.pth 
  
# Age_eval
python -m lib.eval.age_eval \
  --base_dir "outputs/celeba-200" 

# lpips_eval
python -m lib.eval.lpips_eval \
  --folder1 "dataset/eval/agedb-400" \
  --folder2 "outputs/agedb-400" \
  --model_path models/alex.pth
```