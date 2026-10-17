rsync -avz --progress \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  ./ m3:~/bi46_scratch/ncul0003/hons-groundup/
