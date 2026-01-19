#!/bin/bash
# tabpfn_ts experiments for all datasets
# Generated from datasets.yaml; 48 dataset/freq combinations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-tabpfn}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

setup_conda_env() {
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null 2>&1; then
        log_info "Activating existing env: $ENV_NAME"
        conda activate "$ENV_NAME"
    else
        log_info "Creating new env: $ENV_NAME"
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        conda activate "$ENV_NAME"

        log_info "Installing dependencies..."
        pip install tabpfn-time-series
    fi
}

setup_conda_env


########################### Nature ###########################
python experiments/tabpfn_ts.py --dataset "Water_Quality_Darwin/15T"
python experiments/tabpfn_ts.py --dataset "current_velocity/5T"
python experiments/tabpfn_ts.py --dataset "current_velocity/10T"
python experiments/tabpfn_ts.py --dataset "current_velocity/15T"
python experiments/tabpfn_ts.py --dataset "current_velocity/20T"
python experiments/tabpfn_ts.py --dataset "current_velocity/H"
python experiments/tabpfn_ts.py --dataset "CPHL/15T"
python experiments/tabpfn_ts.py --dataset "CPHL/30T"
python experiments/tabpfn_ts.py --dataset "CPHL/H"
python experiments/tabpfn_ts.py --dataset "Coastal_T_S/5T"
python experiments/tabpfn_ts.py --dataset "Coastal_T_S/15T"
python experiments/tabpfn_ts.py --dataset "Coastal_T_S/20T"
python experiments/tabpfn_ts.py --dataset "Coastal_T_S/H"
python experiments/tabpfn_ts.py --dataset "SG_Weather/D"
python experiments/tabpfn_ts.py --dataset "SG_PM25/H"
python experiments/tabpfn_ts.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
python experiments/tabpfn_ts.py --dataset "Australia_Solar/H"
python experiments/tabpfn_ts.py --dataset "epf_electricity_price/H"
python experiments/tabpfn_ts.py --dataset "OpenElectricity_NEM/5T"
python experiments/tabpfn_ts.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
python experiments/tabpfn_ts.py --dataset "SG_Carpark/15T"
python experiments/tabpfn_ts.py --dataset "Finland_Traffic/15T"
python experiments/tabpfn_ts.py --dataset "Port_Activity/D"
python experiments/tabpfn_ts.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
python experiments/tabpfn_ts.py --dataset "ECDC_COVID/D"
python experiments/tabpfn_ts.py --dataset "ECDC_COVID/W"
python experiments/tabpfn_ts.py --dataset "Global_Influenza/W"

########################### Finance ###########################
python experiments/tabpfn_ts.py --dataset "Crypto/D"
python experiments/tabpfn_ts.py --dataset "US_Term_Structure/B"
python experiments/tabpfn_ts.py --dataset "Oil_Price/B"

########################### Economics ###########################
python experiments/tabpfn_ts.py --dataset "Job_Claims/W"
python experiments/tabpfn_ts.py --dataset "Uncertainty_1M/M"
python experiments/tabpfn_ts.py --dataset "Housing_Inventory/M"
python experiments/tabpfn_ts.py --dataset "JOLTS/M"
python experiments/tabpfn_ts.py --dataset "US_Labor/M"
python experiments/tabpfn_ts.py --dataset "Vehicle_Supply/M"
python experiments/tabpfn_ts.py --dataset "Auto_Production_SF/M"
python experiments/tabpfn_ts.py --dataset "Commodity_Production/M"
python experiments/tabpfn_ts.py --dataset "Commodity_Import/M"
python experiments/tabpfn_ts.py --dataset "WUI_Global/Q"
python experiments/tabpfn_ts.py --dataset "Global_Price/Q"

########################### Sales ###########################
python experiments/tabpfn_ts.py --dataset "Vehicle_Sales/M"
python experiments/tabpfn_ts.py --dataset "Online_Retail_2_UCI/D"
python experiments/tabpfn_ts.py --dataset "Supply_Chain_Customer/D"
python experiments/tabpfn_ts.py --dataset "Supply_Chain_Location/D"

########################### CloudOPS ###########################
python experiments/tabpfn_ts.py --dataset "azure2019_D/5T"
python experiments/tabpfn_ts.py --dataset "azure2019_I/5T"
python experiments/tabpfn_ts.py --dataset "azure2019_U/5T"

########################### Industry ###########################
python experiments/tabpfn_ts.py --dataset "Smart_Manufacturing/H"
python experiments/tabpfn_ts.py --dataset "MetroPT-3/5T"
