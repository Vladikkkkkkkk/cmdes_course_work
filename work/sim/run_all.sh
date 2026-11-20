set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../results
python3 experiments.py e1          
python3 experiments.py e2          
python3 verify.py > ../results/verify.log 2>&1   
python3 independent_check.py       
python3 independent_check.py det   
python3 experiments.py e0 e3      
python3 experiments.py e4         
echo "Готово: результати у ../results/"
