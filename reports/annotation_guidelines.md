\# Guidelines d'annotation — Driver Eye State Detection



\## Labels

\- eye\_open : oeil visible et ouvert (meme partiellement)

\- eye\_closed : oeil ferme ou quasi-ferme (moins de 30% ouvert)



\## Regles de decision

\- Oeil mi-clos (30-50% ouvert) : classer eye\_closed si le clignement semble prolonge, eye\_open si cela ressemble a un clignement naturel bref

\- Lunettes : annoter normalement si les yeux restent visibles a travers les verres

\- Reflets/eclairage difficile : annoter au mieux, noter les cas ambigus a part

\- Oeil non visible (angle de tete, main, cheveux) : ne pas annoter cet oeil sur cette frame

\- Un seul oeil visible : annoter uniquement celui-ci



\## Notes

\- Coherence prioritaire sur la precision absolue : appliquer les memes regles sur toutes les frames

