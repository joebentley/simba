(* ::Package:: *)

BeginPackage["Simba`"];


JMatrix::usage="Create diagonal J matrix";
SymplecticJ::usage="Check if given matrix is symplectic under J";
ReorderToPairedForm::usage="Reorder state space to paired operator form";
TransformationToRealisable::usage="Find transformation to physically realisable state space";


Begin["`Private`"];


JMatrix[n_?Positive]:=DiagonalMatrix[ConstantArray[{1,-1},n]//Flatten];
SymplecticJ[tfmatrix_List]:=
Module[{jmat=JMatrix[Dimensions[tfmatrix][[1]]/2]},
ConjugateTranspose[tfmatrix/.s->Conjugate[s]].jmat.(tfmatrix/.s->-s)];

(* Swaps rows e.g. (1, 2, 3, 11, 22, 33)^T \[Rule] (1, 11, 2, 22, 3, 33)^T *)
constructPermutationMatrix[n_?Positive]:=
Module[{m=ConstantArray[0,{2 n,2n}]},
Do[m[[2 i-1,i]]=1;m[[2 i,i+n]]=1,{i,n}];
m];
(* Reorders state-space from (a1, a2, a1\[ConjugateTranspose], a2\[ConjugateTranspose]) \[Rule] (a1, a1\[ConjugateTranspose], a2, a2\[ConjugateTranspose]) *)
ReorderToPairedForm[ss_StateSpaceModel]:=
Module[{a,b,c,d},
{a,b,c,d}=Normal[ss];
Module[{u=constructPermutationMatrix[SystemsModelOrder[ss]/2],
ui=constructPermutationMatrix[SystemsModelDimensions[ss][[1]]/2],
uo=constructPermutationMatrix[SystemsModelDimensions[ss][[2]]/2]},
StateSpaceModel[{u.a.Inverse[u],u.b.Inverse[ui],uo.c.Inverse[u],uo.d.Inverse[ui]}]]];

TransformationToRealisable[ss_StateSpaceModel]:=
Module[{n=SystemsModelOrder@ss,ni,no,ji,a,b,c,d,X},
Clear[x];

{ni,no}=SystemsModelDimensions@ss;
ji=JMatrix[ni/2];
{a,b,c,d}=Normal[ss];
X=Array[x,{n,n}];
Print[Dimensions[X]];
Solve[a.X+X.a\[ConjugateTranspose]+b.ji.b\[ConjugateTranspose]==ConstantArray[0,{n,n}]
&&X.c\[ConjugateTranspose]+b.ji.d\[ConjugateTranspose]==ConstantArray[0,{n,no}]//ComplexExpand,Flatten[X]]];


End[];


EndPackage[];
