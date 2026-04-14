echo 'evaluating mstar (multi query)...'


CHK=mstar_weights/mstar1.pth
VIT=ft_siglip800_hug_cross_rnn

for d in sMQTR wMQTR pMQTR cMQTR 
do
    export IMAGE_EMBED_CACHE="image_cache/${d}"
    rm -r ${IMAGE_EMBED_CACHE}
    mkdir ${IMAGE_EMBED_CACHE}

    python eval/evaluate.py --dataset $d  \
            --rerank  \
            --top_k_ratio 0.05 \
            --device 'cuda:0' \
            --model_name blip2_image_text_matching \
            --vit $VIT \
            --batch_size 4 \
            --text_prompt 'default' \
            --checkpoint ${CHK} \
            --is_conditioned 
done