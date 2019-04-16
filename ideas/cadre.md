# Hypothèses et objectifs

On se place dans le cadre de la détection d'intrusion.
L'objectif du stage est de mettre en place deux architectures : la Défense et l'Attaque.

## La Défense
La Défense est une fonction qui prend en entrée un traffic et qui doit le labeliser comme bénin ou malveillant.


## L'attaque
L'attaque est une fonction qui prend en entrée du bruit et qui doit générer un traffic qui doit être labeliser comme bénin par la Défense



## Le Potentiel
On dispose d'une fonction potentiel qui prend en entrée un traffic et qui lui attribue un score. Cette fonction caractéristique le potentiel de malveillance d'un traffic.
Elle est positive et majorée par 1. Elle est lisse en les variables continues du traffic.



## Hypothèses
- On a un accès illimité aux systèmes 
- Les Faux positifs ont autant d'importance que les Faux Négatifs
- On s'intéresse à un seul type de malveillance à la fois (DOS, probing ...)