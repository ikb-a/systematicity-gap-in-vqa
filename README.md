# Attribute Diversity Determines the Systematicity Gap in VQA

This is the official repository for the paper ["Attribute Diversity Determines the Systematicity Gap in VQA"](https://doi.org/10.18653/v1/2024.emnlp-main.537)

## Download the CLEVR-HOPE dataset

The CLEVR-HOPE dataset is coming to Hugging Face soon! A more compact, 350GB version of CLEVR-HOPE can be downloaded from [the internet archive](https://archive.org/details/CLEVR-HOPE).

## Data generation code

Code and instructions for generating the CLEVR-HOPE dataset can be found under the [./dataset_generation](./dataset_generation) folder. The generation code is different for the minimal and complex splits, and is divided into subfolders. Each subfolder contains a README with detailed instructions.

## Experiment code

Our experiment code and instructions for reproducing our experiments are publicly available at [https://github.com/ikb-a/lxmert-systematicity-gap-in-vqa](https://github.com/ikb-a/lxmert-systematicity-gap-in-vqa)

## Reference

If you've used CLEVR-HOPE in your work, please do cite the paper and let us know by updating the [page for other works using CLEVR-HOPE](FOLLOWUP.md)!

```bibtex
@inproceedings{berlot-attwell-etal-2024-attribute,
    title = "Attribute Diversity Determines the Systematicity Gap in {VQA}",
    author = "Berlot-Attwell, Ian  and
      Agrawal, Kumar Krishna  and
      Carrell, Annabelle Michael  and
      Sharma, Yash  and
      Saphra, Naomi",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.537",
    doi = "10.18653/v1/2024.emnlp-main.537",
    pages = "9576--9611",
    abstract = "Although modern neural networks often generalize to new combinations of familiar concepts, the conditions that enable such compositionality have long been an open question. In this work, we study the systematicity gap in visual question answering: the performance difference between reasoning on previously seen and unseen combinations of object attributes. To test, we introduce a novel diagnostic dataset, CLEVR-HOPE. We find that the systematicity gap is not reduced by increasing the quantity of training data, but is reduced by increasing the diversity of training data. In particular, our experiments suggest that the more distinct attribute type combinations are seen during training, the more systematic we can expect the resulting model to be.",
}
```
