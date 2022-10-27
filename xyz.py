from trankit import Pipeline
from trankit.utils.conll import CoNLL
import trankit
model_dir = "./save_dir/hd"

trankit.verify_customized_pipeline(
        category='customized',
        save_dir= model_dir,
        embedding_name='xlm-roberta-base'
)
p = Pipeline(lang='customized',cache_dir=model_dir)

texts =  ["ಬಗೆಗೆ ದೃಢವಾಗಿ ಯೋಚಿಸಿ ನಿರ್ಣಯವನ್ನು ವಿನಹ ಪ್ರವೃತ್ತಿ ಉಳ್ಳವನು ಆಗಬಾರದು ."
]

def text2conll(text,out_file):
	out = p.posdep(text)
	CoNLL.dict2conll([out['sentences'][0]['tokens']],out_file)
	

print(text2conll(texts[0],'out.conll'))