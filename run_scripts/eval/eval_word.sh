echo 'evaluating mstar (words)...'

CHK=mstar_weights/mstar_word1.pth
VIT=ft_siglip800_word_hug_rnn

for d in CTW
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
            --batch_size 1 \
            --text_prompt 'default' \
            --checkpoint ${CHK}
done