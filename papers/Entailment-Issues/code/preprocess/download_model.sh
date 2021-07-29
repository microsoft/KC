mkdir -p experiments/model
for i in 1 2 3 4 5
do
   git clone https://huggingface.co/kangnichaluo/mnli-$i experiments/model/mnli-$i
done
git clone https://huggingface.co/kangnichaluo/mnli-cb experiments/model/mnli-cb
git clone https://huggingface.co/kangnichaluo/cb experiments/model/cb