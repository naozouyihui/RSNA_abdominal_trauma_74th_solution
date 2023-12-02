cd ../src/Segmentation
echo "### [stage1] Train Segmentation Model : (liver, spleen, left kidney, right kidney, bowel)###"
python train.py

echo "### [stage1] Generate Segmentation Data : (liver, spleen, left kidney, right kidney, bowel)###"
python process_data.py

cd ../Classification_owel
echo "### [stage2] Train Bowel Model : (Bowel) ###"
python train.py

cd ../Classification_kidney
echo "### [stage2] Train Kidney Model : (Kidney) ###"
python train.py

cd ../Classification_liver
echo "### [stage2] Train Liver Model : (Liver) ###"
python train.py

cd ../Classification_spleen
echo "### [stage2] Train Spleen Model : (Spleen) ###"
python train.py

cd ../../bash

