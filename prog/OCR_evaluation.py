from OCR_generic_tools import *

import glob
import re
import json
import sys
import os

def create_str_ner(json_path, str_path):
    """
    Transforme la liste d'entités en une string
    les mots des entités multi-mots sont concaténés
    """
    with open(json_path) as f:
        liste = json.load(f)
    liste = [re.sub("\s", "", x) for x in liste]
    with open(str_path, "w") as w:
        w.write(" ".join(liste))

def get_model_name(chemin):
    model_name = re.split("_", chemin)[-1]
    model_name =re.sub("\.json", "", model_name)
    return model_name

if len(sys.argv)==1:
  print("Donner le chemin des dossiers auteurs en argument (généralement DATA/)")
  exit()

os.makedirs("tmp", exist_ok=True)

path_auteurs = sys.argv[1]
liste_dossiers_auteurs = glob.glob(f"{path_auteurs}/*")
if len(liste_dossiers_auteurs)==0:
  print("Problème with path auteurs: pas de dossier trouvés")
  exit()
#TOD: dans le code d'extraction des entités, créer le nossier NER
for auteur in liste_dossiers_auteurs:
    #if "ADAM" not in auteur:
     #   continue
    print("-"*20)
    #NB: La structure est différente : un level de plus dans le dossier OCR
    reference_files = glob.glob(f"{auteur}/*REF/*.txt")
    ocr_paths = glob.glob(f"{auteur}/*OCR/*/*/*.txt")

    print(re.split("/",auteur)[-1])

    # print("Number of reference files : ", len(reference_files))
    # print("Number of OCR versions : ", len(ocr_paths))

    # print("Number of EN reference files : ",len(en_reference_files))
    # print("Number of EN OCR versions : ",len(en_ocr_paths))

    for reference_file in reference_files:
        # print(ref_file)
        model_name_ref = get_model_name(reference_file)
        print(model_name_ref)
        texte_ref=lire_fichier(reference_file)
        # print(texte_ref)
        for ocr_path in ocr_paths:
            print(ocr_path)
            model_name_ocr = get_model_name(ocr_path)
            configuration = re.split("/", ocr_path)[-1]
            print(configuration)
            sim_path = "/".join(re.split("/", ocr_path)[:-1])+"/SIM/"
            os.makedirs(sim_path, exist_ok=True)
            texte_ocr = lire_fichier(ocr_path)
            # print(texte_ocr)
            distance_txt=get_distances(texte_ref,texte_ocr)
            # print(distance_txt)
            clean_eval_scores_txt = evaluate_file(reference_file, ocr_path)
            clean_eval_scores_txt = {x: y for x, y in clean_eval_scores_txt.items() if "tag" not in x}
            new_scores_text = get_new_scores(texte_ref, texte_ocr)  # TODO:merge
            # TODO: CER, WER
            print(new_scores_text)
            json_path = f"{sim_path}sim2-3_{configuration}.json"
            new_scores_text["clean_eval"] = clean_eval_scores_txt
            for k, v in distance_txt.items():
                new_scores_text[k] = v
            print("  Writing:", json_path)
            with open(json_path, "w") as w:
                w.write(json.dumps(new_scores_text, indent=2))