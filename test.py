from robertagen.model import RoBERTaGen, HFRoBERTaGen, RoBERTaGenConfig

hf_config = RoBERTaGenConfig.from_pretrained('Tianduo/MsAT-RoBERTaGen')
model_args = RoBERTaGen.parse_model_args(hf_config.to_diff_dict())

model = HFRoBERTaGen.from_pretrained(
   'Tianduo/MsAT-RoBERTaGen', 
   config=hf_config, 
   pytorch_model=RoBERTaGen(model_args))