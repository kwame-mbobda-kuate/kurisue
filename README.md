# KURISUE : Kwame-Adam's Urban Road Infrastructure Solver for User Equilibrium
Projet réalisé par Adam Clerget et Kwame Mbobda-Kuate dans le cadre du cours Python pour la data science de deuxième année de l’ENSAE dispensé par Lino Galiana.

## Objectif du projet
Nous étudions le réseau routier de la ville de New York afin d'étudier les chemins empruntés par ses usagers. Autrement dit, en supposant que les conducteurs soient des agents rationnels, à quel point leurs itinéraires, constituant donc un équilibre égoïste, dévie-t-il de l'optimum social ? Nous essayerons de modifier le réseau routier et d'observer les modifications sur les temps de trajets, en espérant mettre en lumière des résultats contre-intuitifs.

## Méthodologie suivie
### La récupération et le traitement des données
Nous nous appuyons sur un jeu de données publié par la Commission des Taxis et des Limousines de New York, détaillant l'intégralité des trajets effectués avec ces modes de transports en 2013. Il s'agit de données public, fournies à la suite d'une demande relative à la loi d'accès à l'information américaine. Le deuxième jeu de données utilisé est issu des enregistreurs de trafic automatisés pour collecter des échantillons de volume de trafic aux croisements de ponts et sur les routes. Il est fourni par le département des transports de la ville de New York.
Les données sont récupérées simplement par des requêtes HTTP. Le premier jeu de données doit être décompressé avant de pouvoir être exploité. Enfin, les données spatiales proviennent d'OpenStreetMaps.

### L’analyse descriptive et la représentation graphique
Nous effectuons quelques tracés de graphes afin d'étudier les tendances des déplacements urbains à diverses échelles temporelles.

### Modélisation
Nous résolvons le problème de l'assignation du traffic dans un cadre statique et uniquement du point de vue de l'équilibre utilisateur. Il s'agit d'un problème classique et nous nous servons du modèle iTAPAS pour le résoudre. Un modèle plus simple et réalisé par nous mêmes mais par la suite abandonné ainsi que la formalisation du problème sont exposés dans le fichier prototype.ipynb. Le modèle iTAPAS est décrit dans [1, 2] et l'implémentation utilisée provient de [3].

## Fonctionnement
Il suffit d'exécuter le fichier main.ipynb. Attention, le programme est très demandant en matière de ressources de stockage (environ 15 Go) et de RAM.

### Références
[1] Xie, J., & Xie, C. (2014, October). An improved TAPAS algorithm for the traffic assignment problem. In Intelligent Transportation Systems (ITSC), 2014 IEEE 17th International Conference on (pp. 2336-2341). IEEE.

[2] Xie, J., & Xie, C. (2016). New insights and improvements of using paired alternative segments for traffic assignment. Transportation Research Part B: Methodological, 93, 406-424.

[3] Han Qiu, (2020). A Python implementation of iTAPAS algorithm for static traffic assignment. https://github.com/hanqiu92/itapas