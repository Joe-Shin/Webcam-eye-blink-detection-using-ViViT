вы!
ь Ќ 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
Ѕ
	AvgPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
÷
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
*
Erf
x"T
y"T"
Ttype:
2
≠
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ј
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	Р
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68дѕ
С
layer_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namelayer_normalization_18/gamma
К
0layer_normalization_18/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_18/gamma*
_output_shapes	
:А*
dtype0
П
layer_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namelayer_normalization_18/beta
И
/layer_normalization_18/beta/Read/ReadVariableOpReadVariableOplayer_normalization_18/beta*
_output_shapes	
:А*
dtype0
С
layer_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namelayer_normalization_19/gamma
К
0layer_normalization_19/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_19/gamma*
_output_shapes	
:А*
dtype0
П
layer_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namelayer_normalization_19/beta
И
/layer_normalization_19/beta/Read/ReadVariableOpReadVariableOplayer_normalization_19/beta*
_output_shapes	
:А*
dtype0
С
layer_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namelayer_normalization_20/gamma
К
0layer_normalization_20/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_20/gamma*
_output_shapes	
:А*
dtype0
П
layer_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namelayer_normalization_20/beta
И
/layer_normalization_20/beta/Read/ReadVariableOpReadVariableOplayer_normalization_20/beta*
_output_shapes	
:А*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	А*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
≤
%tubelet_embedding_32/conv3d_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%tubelet_embedding_32/conv3d_96/kernel
Ђ
9tubelet_embedding_32/conv3d_96/kernel/Read/ReadVariableOpReadVariableOp%tubelet_embedding_32/conv3d_96/kernel**
_output_shapes
: *
dtype0
Ю
#tubelet_embedding_32/conv3d_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#tubelet_embedding_32/conv3d_96/bias
Ч
7tubelet_embedding_32/conv3d_96/bias/Read/ReadVariableOpReadVariableOp#tubelet_embedding_32/conv3d_96/bias*
_output_shapes
: *
dtype0
≤
%tubelet_embedding_32/conv3d_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%tubelet_embedding_32/conv3d_97/kernel
Ђ
9tubelet_embedding_32/conv3d_97/kernel/Read/ReadVariableOpReadVariableOp%tubelet_embedding_32/conv3d_97/kernel**
_output_shapes
: @*
dtype0
Ю
#tubelet_embedding_32/conv3d_97/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#tubelet_embedding_32/conv3d_97/bias
Ч
7tubelet_embedding_32/conv3d_97/bias/Read/ReadVariableOpReadVariableOp#tubelet_embedding_32/conv3d_97/bias*
_output_shapes
:@*
dtype0
≥
%tubelet_embedding_32/conv3d_98/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А*6
shared_name'%tubelet_embedding_32/conv3d_98/kernel
ђ
9tubelet_embedding_32/conv3d_98/kernel/Read/ReadVariableOpReadVariableOp%tubelet_embedding_32/conv3d_98/kernel*+
_output_shapes
:@А*
dtype0
Я
#tubelet_embedding_32/conv3d_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#tubelet_embedding_32/conv3d_98/bias
Ш
7tubelet_embedding_32/conv3d_98/bias/Read/ReadVariableOpReadVariableOp#tubelet_embedding_32/conv3d_98/bias*
_output_shapes	
:А*
dtype0
≤
%tubelet_embedding_33/conv3d_99/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%tubelet_embedding_33/conv3d_99/kernel
Ђ
9tubelet_embedding_33/conv3d_99/kernel/Read/ReadVariableOpReadVariableOp%tubelet_embedding_33/conv3d_99/kernel**
_output_shapes
: *
dtype0
Ю
#tubelet_embedding_33/conv3d_99/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#tubelet_embedding_33/conv3d_99/bias
Ч
7tubelet_embedding_33/conv3d_99/bias/Read/ReadVariableOpReadVariableOp#tubelet_embedding_33/conv3d_99/bias*
_output_shapes
: *
dtype0
і
&tubelet_embedding_33/conv3d_100/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*7
shared_name(&tubelet_embedding_33/conv3d_100/kernel
≠
:tubelet_embedding_33/conv3d_100/kernel/Read/ReadVariableOpReadVariableOp&tubelet_embedding_33/conv3d_100/kernel**
_output_shapes
: @*
dtype0
†
$tubelet_embedding_33/conv3d_100/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$tubelet_embedding_33/conv3d_100/bias
Щ
8tubelet_embedding_33/conv3d_100/bias/Read/ReadVariableOpReadVariableOp$tubelet_embedding_33/conv3d_100/bias*
_output_shapes
:@*
dtype0
µ
&tubelet_embedding_33/conv3d_101/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А*7
shared_name(&tubelet_embedding_33/conv3d_101/kernel
Ѓ
:tubelet_embedding_33/conv3d_101/kernel/Read/ReadVariableOpReadVariableOp&tubelet_embedding_33/conv3d_101/kernel*+
_output_shapes
:@А*
dtype0
°
$tubelet_embedding_33/conv3d_101/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$tubelet_embedding_33/conv3d_101/bias
Ъ
8tubelet_embedding_33/conv3d_101/bias/Read/ReadVariableOpReadVariableOp$tubelet_embedding_33/conv3d_101/bias*
_output_shapes	
:А*
dtype0
±
*positional_encoder_16/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(А*;
shared_name,*positional_encoder_16/embedding/embeddings
™
>positional_encoder_16/embedding/embeddings/Read/ReadVariableOpReadVariableOp*positional_encoder_16/embedding/embeddings*
_output_shapes
:	(А*
dtype0
®
#multi_head_attention_6/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#multi_head_attention_6/query/kernel
°
7multi_head_attention_6/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_6/query/kernel*$
_output_shapes
:АА*
dtype0
Я
!multi_head_attention_6/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*2
shared_name#!multi_head_attention_6/query/bias
Ш
5multi_head_attention_6/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/query/bias*
_output_shapes
:	А*
dtype0
§
!multi_head_attention_6/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*2
shared_name#!multi_head_attention_6/key/kernel
Э
5multi_head_attention_6/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/key/kernel*$
_output_shapes
:АА*
dtype0
Ы
multi_head_attention_6/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*0
shared_name!multi_head_attention_6/key/bias
Ф
3multi_head_attention_6/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_6/key/bias*
_output_shapes
:	А*
dtype0
®
#multi_head_attention_6/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*4
shared_name%#multi_head_attention_6/value/kernel
°
7multi_head_attention_6/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_6/value/kernel*$
_output_shapes
:АА*
dtype0
Я
!multi_head_attention_6/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*2
shared_name#!multi_head_attention_6/value/bias
Ш
5multi_head_attention_6/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/value/bias*
_output_shapes
:	А*
dtype0
Њ
.multi_head_attention_6/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*?
shared_name0.multi_head_attention_6/attention_output/kernel
Ј
Bmulti_head_attention_6/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_6/attention_output/kernel*$
_output_shapes
:АА*
dtype0
±
,multi_head_attention_6/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*=
shared_name.,multi_head_attention_6/attention_output/bias
™
@multi_head_attention_6/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_6/attention_output/bias*
_output_shapes	
:А*
dtype0
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:А*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Я
#Adam/layer_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/layer_normalization_18/gamma/m
Ш
7Adam/layer_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_18/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/layer_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/layer_normalization_18/beta/m
Ц
6Adam/layer_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_18/beta/m*
_output_shapes	
:А*
dtype0
Я
#Adam/layer_normalization_19/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/layer_normalization_19/gamma/m
Ш
7Adam/layer_normalization_19/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_19/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/layer_normalization_19/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/layer_normalization_19/beta/m
Ц
6Adam/layer_normalization_19/beta/m/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_19/beta/m*
_output_shapes	
:А*
dtype0
Я
#Adam/layer_normalization_20/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/layer_normalization_20/gamma/m
Ш
7Adam/layer_normalization_20/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_20/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/layer_normalization_20/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/layer_normalization_20/beta/m
Ц
6Adam/layer_normalization_20/beta/m/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_20/beta/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_13/kernel/m
В
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes
:	А*
dtype0
А
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0
ј
,Adam/tubelet_embedding_32/conv3d_96/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/tubelet_embedding_32/conv3d_96/kernel/m
є
@Adam/tubelet_embedding_32/conv3d_96/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_32/conv3d_96/kernel/m**
_output_shapes
: *
dtype0
ђ
*Adam/tubelet_embedding_32/conv3d_96/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/tubelet_embedding_32/conv3d_96/bias/m
•
>Adam/tubelet_embedding_32/conv3d_96/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_32/conv3d_96/bias/m*
_output_shapes
: *
dtype0
ј
,Adam/tubelet_embedding_32/conv3d_97/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/tubelet_embedding_32/conv3d_97/kernel/m
є
@Adam/tubelet_embedding_32/conv3d_97/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_32/conv3d_97/kernel/m**
_output_shapes
: @*
dtype0
ђ
*Adam/tubelet_embedding_32/conv3d_97/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/tubelet_embedding_32/conv3d_97/bias/m
•
>Adam/tubelet_embedding_32/conv3d_97/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_32/conv3d_97/bias/m*
_output_shapes
:@*
dtype0
Ѕ
,Adam/tubelet_embedding_32/conv3d_98/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А*=
shared_name.,Adam/tubelet_embedding_32/conv3d_98/kernel/m
Ї
@Adam/tubelet_embedding_32/conv3d_98/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_32/conv3d_98/kernel/m*+
_output_shapes
:@А*
dtype0
≠
*Adam/tubelet_embedding_32/conv3d_98/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*;
shared_name,*Adam/tubelet_embedding_32/conv3d_98/bias/m
¶
>Adam/tubelet_embedding_32/conv3d_98/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_32/conv3d_98/bias/m*
_output_shapes	
:А*
dtype0
ј
,Adam/tubelet_embedding_33/conv3d_99/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/tubelet_embedding_33/conv3d_99/kernel/m
є
@Adam/tubelet_embedding_33/conv3d_99/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_33/conv3d_99/kernel/m**
_output_shapes
: *
dtype0
ђ
*Adam/tubelet_embedding_33/conv3d_99/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/tubelet_embedding_33/conv3d_99/bias/m
•
>Adam/tubelet_embedding_33/conv3d_99/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_33/conv3d_99/bias/m*
_output_shapes
: *
dtype0
¬
-Adam/tubelet_embedding_33/conv3d_100/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*>
shared_name/-Adam/tubelet_embedding_33/conv3d_100/kernel/m
ї
AAdam/tubelet_embedding_33/conv3d_100/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/tubelet_embedding_33/conv3d_100/kernel/m**
_output_shapes
: @*
dtype0
Ѓ
+Adam/tubelet_embedding_33/conv3d_100/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/tubelet_embedding_33/conv3d_100/bias/m
І
?Adam/tubelet_embedding_33/conv3d_100/bias/m/Read/ReadVariableOpReadVariableOp+Adam/tubelet_embedding_33/conv3d_100/bias/m*
_output_shapes
:@*
dtype0
√
-Adam/tubelet_embedding_33/conv3d_101/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А*>
shared_name/-Adam/tubelet_embedding_33/conv3d_101/kernel/m
Љ
AAdam/tubelet_embedding_33/conv3d_101/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/tubelet_embedding_33/conv3d_101/kernel/m*+
_output_shapes
:@А*
dtype0
ѓ
+Adam/tubelet_embedding_33/conv3d_101/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/tubelet_embedding_33/conv3d_101/bias/m
®
?Adam/tubelet_embedding_33/conv3d_101/bias/m/Read/ReadVariableOpReadVariableOp+Adam/tubelet_embedding_33/conv3d_101/bias/m*
_output_shapes	
:А*
dtype0
њ
1Adam/positional_encoder_16/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(А*B
shared_name31Adam/positional_encoder_16/embedding/embeddings/m
Є
EAdam/positional_encoder_16/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp1Adam/positional_encoder_16/embedding/embeddings/m*
_output_shapes
:	(А*
dtype0
ґ
*Adam/multi_head_attention_6/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*;
shared_name,*Adam/multi_head_attention_6/query/kernel/m
ѓ
>Adam/multi_head_attention_6/query/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_6/query/kernel/m*$
_output_shapes
:АА*
dtype0
≠
(Adam/multi_head_attention_6/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*9
shared_name*(Adam/multi_head_attention_6/query/bias/m
¶
<Adam/multi_head_attention_6/query/bias/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_6/query/bias/m*
_output_shapes
:	А*
dtype0
≤
(Adam/multi_head_attention_6/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*9
shared_name*(Adam/multi_head_attention_6/key/kernel/m
Ђ
<Adam/multi_head_attention_6/key/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_6/key/kernel/m*$
_output_shapes
:АА*
dtype0
©
&Adam/multi_head_attention_6/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*7
shared_name(&Adam/multi_head_attention_6/key/bias/m
Ґ
:Adam/multi_head_attention_6/key/bias/m/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention_6/key/bias/m*
_output_shapes
:	А*
dtype0
ґ
*Adam/multi_head_attention_6/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*;
shared_name,*Adam/multi_head_attention_6/value/kernel/m
ѓ
>Adam/multi_head_attention_6/value/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_6/value/kernel/m*$
_output_shapes
:АА*
dtype0
≠
(Adam/multi_head_attention_6/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*9
shared_name*(Adam/multi_head_attention_6/value/bias/m
¶
<Adam/multi_head_attention_6/value/bias/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_6/value/bias/m*
_output_shapes
:	А*
dtype0
ћ
5Adam/multi_head_attention_6/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*F
shared_name75Adam/multi_head_attention_6/attention_output/kernel/m
≈
IAdam/multi_head_attention_6/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/multi_head_attention_6/attention_output/kernel/m*$
_output_shapes
:АА*
dtype0
њ
3Adam/multi_head_attention_6/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*D
shared_name53Adam/multi_head_attention_6/attention_output/bias/m
Є
GAdam/multi_head_attention_6/attention_output/bias/m/Read/ReadVariableOpReadVariableOp3Adam/multi_head_attention_6/attention_output/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_12/kernel/m
Г
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_12/bias/m
z
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes	
:А*
dtype0
Я
#Adam/layer_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/layer_normalization_18/gamma/v
Ш
7Adam/layer_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_18/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/layer_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/layer_normalization_18/beta/v
Ц
6Adam/layer_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_18/beta/v*
_output_shapes	
:А*
dtype0
Я
#Adam/layer_normalization_19/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/layer_normalization_19/gamma/v
Ш
7Adam/layer_normalization_19/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_19/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/layer_normalization_19/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/layer_normalization_19/beta/v
Ц
6Adam/layer_normalization_19/beta/v/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_19/beta/v*
_output_shapes	
:А*
dtype0
Я
#Adam/layer_normalization_20/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/layer_normalization_20/gamma/v
Ш
7Adam/layer_normalization_20/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_20/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/layer_normalization_20/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/layer_normalization_20/beta/v
Ц
6Adam/layer_normalization_20/beta/v/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_20/beta/v*
_output_shapes	
:А*
dtype0
Й
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_13/kernel/v
В
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes
:	А*
dtype0
А
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0
ј
,Adam/tubelet_embedding_32/conv3d_96/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/tubelet_embedding_32/conv3d_96/kernel/v
є
@Adam/tubelet_embedding_32/conv3d_96/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_32/conv3d_96/kernel/v**
_output_shapes
: *
dtype0
ђ
*Adam/tubelet_embedding_32/conv3d_96/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/tubelet_embedding_32/conv3d_96/bias/v
•
>Adam/tubelet_embedding_32/conv3d_96/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_32/conv3d_96/bias/v*
_output_shapes
: *
dtype0
ј
,Adam/tubelet_embedding_32/conv3d_97/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/tubelet_embedding_32/conv3d_97/kernel/v
є
@Adam/tubelet_embedding_32/conv3d_97/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_32/conv3d_97/kernel/v**
_output_shapes
: @*
dtype0
ђ
*Adam/tubelet_embedding_32/conv3d_97/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/tubelet_embedding_32/conv3d_97/bias/v
•
>Adam/tubelet_embedding_32/conv3d_97/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_32/conv3d_97/bias/v*
_output_shapes
:@*
dtype0
Ѕ
,Adam/tubelet_embedding_32/conv3d_98/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А*=
shared_name.,Adam/tubelet_embedding_32/conv3d_98/kernel/v
Ї
@Adam/tubelet_embedding_32/conv3d_98/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_32/conv3d_98/kernel/v*+
_output_shapes
:@А*
dtype0
≠
*Adam/tubelet_embedding_32/conv3d_98/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*;
shared_name,*Adam/tubelet_embedding_32/conv3d_98/bias/v
¶
>Adam/tubelet_embedding_32/conv3d_98/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_32/conv3d_98/bias/v*
_output_shapes	
:А*
dtype0
ј
,Adam/tubelet_embedding_33/conv3d_99/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/tubelet_embedding_33/conv3d_99/kernel/v
є
@Adam/tubelet_embedding_33/conv3d_99/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_33/conv3d_99/kernel/v**
_output_shapes
: *
dtype0
ђ
*Adam/tubelet_embedding_33/conv3d_99/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/tubelet_embedding_33/conv3d_99/bias/v
•
>Adam/tubelet_embedding_33/conv3d_99/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_33/conv3d_99/bias/v*
_output_shapes
: *
dtype0
¬
-Adam/tubelet_embedding_33/conv3d_100/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*>
shared_name/-Adam/tubelet_embedding_33/conv3d_100/kernel/v
ї
AAdam/tubelet_embedding_33/conv3d_100/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/tubelet_embedding_33/conv3d_100/kernel/v**
_output_shapes
: @*
dtype0
Ѓ
+Adam/tubelet_embedding_33/conv3d_100/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adam/tubelet_embedding_33/conv3d_100/bias/v
І
?Adam/tubelet_embedding_33/conv3d_100/bias/v/Read/ReadVariableOpReadVariableOp+Adam/tubelet_embedding_33/conv3d_100/bias/v*
_output_shapes
:@*
dtype0
√
-Adam/tubelet_embedding_33/conv3d_101/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А*>
shared_name/-Adam/tubelet_embedding_33/conv3d_101/kernel/v
Љ
AAdam/tubelet_embedding_33/conv3d_101/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/tubelet_embedding_33/conv3d_101/kernel/v*+
_output_shapes
:@А*
dtype0
ѓ
+Adam/tubelet_embedding_33/conv3d_101/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*<
shared_name-+Adam/tubelet_embedding_33/conv3d_101/bias/v
®
?Adam/tubelet_embedding_33/conv3d_101/bias/v/Read/ReadVariableOpReadVariableOp+Adam/tubelet_embedding_33/conv3d_101/bias/v*
_output_shapes	
:А*
dtype0
њ
1Adam/positional_encoder_16/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(А*B
shared_name31Adam/positional_encoder_16/embedding/embeddings/v
Є
EAdam/positional_encoder_16/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp1Adam/positional_encoder_16/embedding/embeddings/v*
_output_shapes
:	(А*
dtype0
ґ
*Adam/multi_head_attention_6/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*;
shared_name,*Adam/multi_head_attention_6/query/kernel/v
ѓ
>Adam/multi_head_attention_6/query/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_6/query/kernel/v*$
_output_shapes
:АА*
dtype0
≠
(Adam/multi_head_attention_6/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*9
shared_name*(Adam/multi_head_attention_6/query/bias/v
¶
<Adam/multi_head_attention_6/query/bias/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_6/query/bias/v*
_output_shapes
:	А*
dtype0
≤
(Adam/multi_head_attention_6/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*9
shared_name*(Adam/multi_head_attention_6/key/kernel/v
Ђ
<Adam/multi_head_attention_6/key/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_6/key/kernel/v*$
_output_shapes
:АА*
dtype0
©
&Adam/multi_head_attention_6/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*7
shared_name(&Adam/multi_head_attention_6/key/bias/v
Ґ
:Adam/multi_head_attention_6/key/bias/v/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention_6/key/bias/v*
_output_shapes
:	А*
dtype0
ґ
*Adam/multi_head_attention_6/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*;
shared_name,*Adam/multi_head_attention_6/value/kernel/v
ѓ
>Adam/multi_head_attention_6/value/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_6/value/kernel/v*$
_output_shapes
:АА*
dtype0
≠
(Adam/multi_head_attention_6/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*9
shared_name*(Adam/multi_head_attention_6/value/bias/v
¶
<Adam/multi_head_attention_6/value/bias/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_6/value/bias/v*
_output_shapes
:	А*
dtype0
ћ
5Adam/multi_head_attention_6/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*F
shared_name75Adam/multi_head_attention_6/attention_output/kernel/v
≈
IAdam/multi_head_attention_6/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/multi_head_attention_6/attention_output/kernel/v*$
_output_shapes
:АА*
dtype0
њ
3Adam/multi_head_attention_6/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*D
shared_name53Adam/multi_head_attention_6/attention_output/bias/v
Є
GAdam/multi_head_attention_6/attention_output/bias/v/Read/ReadVariableOpReadVariableOp3Adam/multi_head_attention_6/attention_output/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_12/kernel/v
Г
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_12/bias/v
z
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes	
:А*
dtype0
т
ConstConst*
_output_shapes
:(*
dtype0*Є
valueЃBЂ("†                            	   
                                                                      !   "   #   $   %   &   '   

NoOpNoOp
ьК
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*іК
value©КB•К BЭК
М
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 

	keras_api* 

	keras_api* 
п

projection
pool
projection2
	pool2
 projection3
	!pool4
"flatten
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses*
п
)
projection
*pool
+projection2
	,pool2
-projection3
	.pool4
/flatten
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*
О
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
®
<position_embedding
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
ѓ
Caxis
	Dgamma
Ebeta
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*
щ
L_query_dense
M
_key_dense
N_value_dense
O_softmax
P_dropout_layer
Q_output_dense
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses*
О
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 
ѓ
^axis
	_gamma
`beta
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses*
Ј
glayer_with_weights-0
glayer-0
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses*
О
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 
ѓ
taxis
	ugamma
vbeta
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses*
С
}	variables
~trainable_variables
regularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses* 
Ѓ
Гkernel
	Дbias
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses*
з
	Лiter
Мbeta_1
Нbeta_2

Оdecay
Пlearning_rateDmфEmх_mц`mчumшvmщ	Гmъ	Дmы	Рmь	Сmэ	Тmю	Уm€	ФmА	ХmБ	ЦmВ	ЧmГ	ШmД	ЩmЕ	ЪmЖ	ЫmЗ	ЬmИ	ЭmЙ	ЮmК	ЯmЛ	†mМ	°mН	ҐmО	£mП	§mР	•mС	¶mТDvУEvФ_vХ`vЦuvЧvvШ	ГvЩ	ДvЪ	РvЫ	СvЬ	ТvЭ	УvЮ	ФvЯ	Хv†	Цv°	ЧvҐ	Шv£	Щv§	Ъv•	Ыv¶	ЬvІ	Эv®	Юv©	Яv™	†vЂ	°vђ	Ґv≠	£vЃ	§vѓ	•v∞	¶v±*
Л
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
Ш8
Щ9
Ъ10
Ы11
Ь12
D13
E14
Э15
Ю16
Я17
†18
°19
Ґ20
£21
§22
_23
`24
•25
¶26
u27
v28
Г29
Д30*
Л
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
Ш8
Щ9
Ъ10
Ы11
Ь12
D13
E14
Э15
Ю16
Я17
†18
°19
Ґ20
£21
§22
_23
`24
•25
¶26
u27
v28
Г29
Д30*
* 
µ
Іnon_trainable_variables
®layers
©metrics
 ™layer_regularization_losses
Ђlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

ђserving_default* 
* 
* 
Ѓ
Рkernel
	Сbias
≠	variables
Ѓtrainable_variables
ѓregularization_losses
∞	keras_api
±__call__
+≤&call_and_return_all_conditional_losses*
Ф
≥	variables
іtrainable_variables
µregularization_losses
ґ	keras_api
Ј__call__
+Є&call_and_return_all_conditional_losses* 
Ѓ
Тkernel
	Уbias
є	variables
Їtrainable_variables
їregularization_losses
Љ	keras_api
љ__call__
+Њ&call_and_return_all_conditional_losses*
Ф
њ	variables
јtrainable_variables
Ѕregularization_losses
¬	keras_api
√__call__
+ƒ&call_and_return_all_conditional_losses* 
Ѓ
Фkernel
	Хbias
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
…__call__
+ &call_and_return_all_conditional_losses*
Ф
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ќ	keras_api
ѕ__call__
+–&call_and_return_all_conditional_losses* 
Ф
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses* 
4
Р0
С1
Т2
У3
Ф4
Х5*
4
Р0
С1
Т2
У3
Ф4
Х5*
* 
Ш
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
* 
* 
Ѓ
Цkernel
	Чbias
№	variables
Ёtrainable_variables
ёregularization_losses
я	keras_api
а__call__
+б&call_and_return_all_conditional_losses*
Ф
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses* 
Ѓ
Шkernel
	Щbias
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses*
Ф
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses* 
Ѓ
Ъkernel
	Ыbias
ф	variables
хtrainable_variables
цregularization_losses
ч	keras_api
ш__call__
+щ&call_and_return_all_conditional_losses*
Ф
ъ	variables
ыtrainable_variables
ьregularization_losses
э	keras_api
ю__call__
+€&call_and_return_all_conditional_losses* 
Ф
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses* 
4
Ц0
Ч1
Ш2
Щ3
Ъ4
Ы5*
4
Ц0
Ч1
Ш2
Щ3
Ъ4
Ы5*
* 
Ш
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
І
Ь
embeddings
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses*

Ь0*

Ь0*
* 
Ш
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUElayer_normalization_18/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_18/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*
* 
Ш
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
б
†partial_output_shape
°full_output_shape
Эkernel
	Юbias
Ґ	variables
£trainable_variables
§regularization_losses
•	keras_api
¶__call__
+І&call_and_return_all_conditional_losses*
б
®partial_output_shape
©full_output_shape
Яkernel
	†bias
™	variables
Ђtrainable_variables
ђregularization_losses
≠	keras_api
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses*
б
∞partial_output_shape
±full_output_shape
°kernel
	Ґbias
≤	variables
≥trainable_variables
іregularization_losses
µ	keras_api
ґ__call__
+Ј&call_and_return_all_conditional_losses*
Ф
Є	variables
єtrainable_variables
Їregularization_losses
ї	keras_api
Љ__call__
+љ&call_and_return_all_conditional_losses* 
ђ
Њ	variables
њtrainable_variables
јregularization_losses
Ѕ	keras_api
¬_random_generator
√__call__
+ƒ&call_and_return_all_conditional_losses* 
б
≈partial_output_shape
∆full_output_shape
£kernel
	§bias
«	variables
»trainable_variables
…regularization_losses
 	keras_api
Ћ__call__
+ћ&call_and_return_all_conditional_losses*
D
Э0
Ю1
Я2
†3
°4
Ґ5
£6
§7*
D
Э0
Ю1
Я2
†3
°4
Ґ5
£6
§7*
* 
Ш
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 
* 
* 
* 
ke
VARIABLE_VALUElayer_normalization_19/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_19/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*

_0
`1*

_0
`1*
* 
Ш
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
Ѓ
•kernel
	¶bias
№	variables
Ёtrainable_variables
ёregularization_losses
я	keras_api
а__call__
+б&call_and_return_all_conditional_losses*

•0
¶1*

•0
¶1*
* 
Ш
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 
* 
* 
* 
ke
VARIABLE_VALUElayer_normalization_20/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_20/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

u0
v1*
* 
Ш
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Щ
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
}	variables
~trainable_variables
regularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_13/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

Г0
Д1*

Г0
Д1*
* 
Ю
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%tubelet_embedding_32/conv3d_96/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#tubelet_embedding_32/conv3d_96/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%tubelet_embedding_32/conv3d_97/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#tubelet_embedding_32/conv3d_97/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%tubelet_embedding_32/conv3d_98/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#tubelet_embedding_32/conv3d_98/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%tubelet_embedding_33/conv3d_99/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#tubelet_embedding_33/conv3d_99/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&tubelet_embedding_33/conv3d_100/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$tubelet_embedding_33/conv3d_100/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&tubelet_embedding_33/conv3d_101/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tubelet_embedding_33/conv3d_101/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*positional_encoder_16/embedding/embeddings'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_6/query/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_6/query/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_6/key/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_6/key/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_6/value/kernel'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_6/value/bias'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_6/attention_output/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_6/attention_output/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_12/kernel'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_12/bias'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
* 
z
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15*

ы0
ь1*
* 
* 
* 

Р0
С1*

Р0
С1*
* 
Ю
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
≠	variables
Ѓtrainable_variables
ѓregularization_losses
±__call__
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
≥	variables
іtrainable_variables
µregularization_losses
Ј__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses* 
* 
* 

Т0
У1*

Т0
У1*
* 
Ю
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
є	variables
Їtrainable_variables
їregularization_losses
љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
њ	variables
јtrainable_variables
Ѕregularization_losses
√__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses* 
* 
* 

Ф0
Х1*

Ф0
Х1*
* 
Ю
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
≈	variables
∆trainable_variables
«regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ѕ__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses* 
* 
* 
* 
5
0
1
2
3
 4
!5
"6*
* 
* 
* 

Ц0
Ч1*

Ц0
Ч1*
* 
Ю
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
№	variables
Ёtrainable_variables
ёregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses* 
* 
* 

Ш0
Щ1*

Ш0
Щ1*
* 
Ю
™non_trainable_variables
Ђlayers
ђmetrics
 ≠layer_regularization_losses
Ѓlayer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
ѓnon_trainable_variables
∞layers
±metrics
 ≤layer_regularization_losses
≥layer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses* 
* 
* 

Ъ0
Ы1*

Ъ0
Ы1*
* 
Ю
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
ф	variables
хtrainable_variables
цregularization_losses
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
ъ	variables
ыtrainable_variables
ьregularization_losses
ю__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses* 
* 
* 
* 
5
)0
*1
+2
,3
-4
.5
/6*
* 
* 
* 
* 
* 
* 
* 
* 

Ь0*

Ь0*
* 
Ю
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses*
* 
* 
* 

<0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Э0
Ю1*

Э0
Ю1*
* 
Ю
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
Ґ	variables
£trainable_variables
§regularization_losses
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses*
* 
* 
* 
* 

Я0
†1*

Я0
†1*
* 
Ю
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
™	variables
Ђtrainable_variables
ђregularization_losses
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses*
* 
* 
* 
* 

°0
Ґ1*

°0
Ґ1*
* 
Ю
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
≤	variables
≥trainable_variables
іregularization_losses
ґ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
Є	variables
єtrainable_variables
Їregularization_losses
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
№non_trainable_variables
Ёlayers
ёmetrics
 яlayer_regularization_losses
аlayer_metrics
Њ	variables
њtrainable_variables
јregularization_losses
√__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

£0
§1*

£0
§1*
* 
Ю
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
«	variables
»trainable_variables
…regularization_losses
Ћ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses*
* 
* 
* 
.
L0
M1
N2
O3
P4
Q5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

•0
¶1*

•0
¶1*
* 
Ю
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
№	variables
Ёtrainable_variables
ёregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses*
* 
* 
* 

g0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

лtotal

мcount
н	variables
о	keras_api*
M

пtotal

рcount
с
_fn_kwargs
т	variables
у	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

л0
м1*

н	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

п0
р1*

т	variables*
ПИ
VARIABLE_VALUE#Adam/layer_normalization_18/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/layer_normalization_18/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/layer_normalization_19/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/layer_normalization_19/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/layer_normalization_20/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/layer_normalization_20/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/tubelet_embedding_32/conv3d_96/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/tubelet_embedding_32/conv3d_96/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/tubelet_embedding_32/conv3d_97/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/tubelet_embedding_32/conv3d_97/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/tubelet_embedding_32/conv3d_98/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/tubelet_embedding_32/conv3d_98/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/tubelet_embedding_33/conv3d_99/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/tubelet_embedding_33/conv3d_99/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUE-Adam/tubelet_embedding_33/conv3d_100/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUE+Adam/tubelet_embedding_33/conv3d_100/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE-Adam/tubelet_embedding_33/conv3d_101/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE+Adam/tubelet_embedding_33/conv3d_101/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE1Adam/positional_encoder_16/embedding/embeddings/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUE*Adam/multi_head_attention_6/query/kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_6/query/bias/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_6/key/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/multi_head_attention_6/key/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUE*Adam/multi_head_attention_6/value/kernel/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_6/value/bias/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE5Adam/multi_head_attention_6/attention_output/kernel/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE3Adam/multi_head_attention_6/attention_output/bias/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_12/kernel/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_12/bias/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/layer_normalization_18/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/layer_normalization_18/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/layer_normalization_19/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/layer_normalization_19/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/layer_normalization_20/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/layer_normalization_20/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/tubelet_embedding_32/conv3d_96/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/tubelet_embedding_32/conv3d_96/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/tubelet_embedding_32/conv3d_97/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/tubelet_embedding_32/conv3d_97/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/tubelet_embedding_32/conv3d_98/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/tubelet_embedding_32/conv3d_98/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/tubelet_embedding_33/conv3d_99/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUE*Adam/tubelet_embedding_33/conv3d_99/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUE-Adam/tubelet_embedding_33/conv3d_100/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUE+Adam/tubelet_embedding_33/conv3d_100/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUE-Adam/tubelet_embedding_33/conv3d_101/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE+Adam/tubelet_embedding_33/conv3d_101/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE1Adam/positional_encoder_16/embedding/embeddings/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUE*Adam/multi_head_attention_6/query/kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_6/query/bias/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_6/key/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/multi_head_attention_6/key/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUE*Adam/multi_head_attention_6/value/kernel/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_6/value/bias/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE5Adam/multi_head_attention_6/attention_output/kernel/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE3Adam/multi_head_attention_6/attention_output/bias/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_12/kernel/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_12/bias/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
У
serving_default_input_17Placeholder*3
_output_shapes!
:€€€€€€€€€
*
dtype0*(
shape:€€€€€€€€€

Щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_17%tubelet_embedding_32/conv3d_96/kernel#tubelet_embedding_32/conv3d_96/bias%tubelet_embedding_32/conv3d_97/kernel#tubelet_embedding_32/conv3d_97/bias%tubelet_embedding_32/conv3d_98/kernel#tubelet_embedding_32/conv3d_98/bias%tubelet_embedding_33/conv3d_99/kernel#tubelet_embedding_33/conv3d_99/bias&tubelet_embedding_33/conv3d_100/kernel$tubelet_embedding_33/conv3d_100/bias&tubelet_embedding_33/conv3d_101/kernel$tubelet_embedding_33/conv3d_101/biasConst*positional_encoder_16/embedding/embeddingslayer_normalization_18/gammalayer_normalization_18/beta#multi_head_attention_6/query/kernel!multi_head_attention_6/query/bias!multi_head_attention_6/key/kernelmulti_head_attention_6/key/bias#multi_head_attention_6/value/kernel!multi_head_attention_6/value/bias.multi_head_attention_6/attention_output/kernel,multi_head_attention_6/attention_output/biaslayer_normalization_19/gammalayer_normalization_19/betadense_12/kerneldense_12/biaslayer_normalization_20/gammalayer_normalization_20/betadense_13/kerneldense_13/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8В *-
f(R&
$__inference_signature_wrapper_496102
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
и/
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0layer_normalization_18/gamma/Read/ReadVariableOp/layer_normalization_18/beta/Read/ReadVariableOp0layer_normalization_19/gamma/Read/ReadVariableOp/layer_normalization_19/beta/Read/ReadVariableOp0layer_normalization_20/gamma/Read/ReadVariableOp/layer_normalization_20/beta/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9tubelet_embedding_32/conv3d_96/kernel/Read/ReadVariableOp7tubelet_embedding_32/conv3d_96/bias/Read/ReadVariableOp9tubelet_embedding_32/conv3d_97/kernel/Read/ReadVariableOp7tubelet_embedding_32/conv3d_97/bias/Read/ReadVariableOp9tubelet_embedding_32/conv3d_98/kernel/Read/ReadVariableOp7tubelet_embedding_32/conv3d_98/bias/Read/ReadVariableOp9tubelet_embedding_33/conv3d_99/kernel/Read/ReadVariableOp7tubelet_embedding_33/conv3d_99/bias/Read/ReadVariableOp:tubelet_embedding_33/conv3d_100/kernel/Read/ReadVariableOp8tubelet_embedding_33/conv3d_100/bias/Read/ReadVariableOp:tubelet_embedding_33/conv3d_101/kernel/Read/ReadVariableOp8tubelet_embedding_33/conv3d_101/bias/Read/ReadVariableOp>positional_encoder_16/embedding/embeddings/Read/ReadVariableOp7multi_head_attention_6/query/kernel/Read/ReadVariableOp5multi_head_attention_6/query/bias/Read/ReadVariableOp5multi_head_attention_6/key/kernel/Read/ReadVariableOp3multi_head_attention_6/key/bias/Read/ReadVariableOp7multi_head_attention_6/value/kernel/Read/ReadVariableOp5multi_head_attention_6/value/bias/Read/ReadVariableOpBmulti_head_attention_6/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_6/attention_output/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adam/layer_normalization_18/gamma/m/Read/ReadVariableOp6Adam/layer_normalization_18/beta/m/Read/ReadVariableOp7Adam/layer_normalization_19/gamma/m/Read/ReadVariableOp6Adam/layer_normalization_19/beta/m/Read/ReadVariableOp7Adam/layer_normalization_20/gamma/m/Read/ReadVariableOp6Adam/layer_normalization_20/beta/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp@Adam/tubelet_embedding_32/conv3d_96/kernel/m/Read/ReadVariableOp>Adam/tubelet_embedding_32/conv3d_96/bias/m/Read/ReadVariableOp@Adam/tubelet_embedding_32/conv3d_97/kernel/m/Read/ReadVariableOp>Adam/tubelet_embedding_32/conv3d_97/bias/m/Read/ReadVariableOp@Adam/tubelet_embedding_32/conv3d_98/kernel/m/Read/ReadVariableOp>Adam/tubelet_embedding_32/conv3d_98/bias/m/Read/ReadVariableOp@Adam/tubelet_embedding_33/conv3d_99/kernel/m/Read/ReadVariableOp>Adam/tubelet_embedding_33/conv3d_99/bias/m/Read/ReadVariableOpAAdam/tubelet_embedding_33/conv3d_100/kernel/m/Read/ReadVariableOp?Adam/tubelet_embedding_33/conv3d_100/bias/m/Read/ReadVariableOpAAdam/tubelet_embedding_33/conv3d_101/kernel/m/Read/ReadVariableOp?Adam/tubelet_embedding_33/conv3d_101/bias/m/Read/ReadVariableOpEAdam/positional_encoder_16/embedding/embeddings/m/Read/ReadVariableOp>Adam/multi_head_attention_6/query/kernel/m/Read/ReadVariableOp<Adam/multi_head_attention_6/query/bias/m/Read/ReadVariableOp<Adam/multi_head_attention_6/key/kernel/m/Read/ReadVariableOp:Adam/multi_head_attention_6/key/bias/m/Read/ReadVariableOp>Adam/multi_head_attention_6/value/kernel/m/Read/ReadVariableOp<Adam/multi_head_attention_6/value/bias/m/Read/ReadVariableOpIAdam/multi_head_attention_6/attention_output/kernel/m/Read/ReadVariableOpGAdam/multi_head_attention_6/attention_output/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp7Adam/layer_normalization_18/gamma/v/Read/ReadVariableOp6Adam/layer_normalization_18/beta/v/Read/ReadVariableOp7Adam/layer_normalization_19/gamma/v/Read/ReadVariableOp6Adam/layer_normalization_19/beta/v/Read/ReadVariableOp7Adam/layer_normalization_20/gamma/v/Read/ReadVariableOp6Adam/layer_normalization_20/beta/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp@Adam/tubelet_embedding_32/conv3d_96/kernel/v/Read/ReadVariableOp>Adam/tubelet_embedding_32/conv3d_96/bias/v/Read/ReadVariableOp@Adam/tubelet_embedding_32/conv3d_97/kernel/v/Read/ReadVariableOp>Adam/tubelet_embedding_32/conv3d_97/bias/v/Read/ReadVariableOp@Adam/tubelet_embedding_32/conv3d_98/kernel/v/Read/ReadVariableOp>Adam/tubelet_embedding_32/conv3d_98/bias/v/Read/ReadVariableOp@Adam/tubelet_embedding_33/conv3d_99/kernel/v/Read/ReadVariableOp>Adam/tubelet_embedding_33/conv3d_99/bias/v/Read/ReadVariableOpAAdam/tubelet_embedding_33/conv3d_100/kernel/v/Read/ReadVariableOp?Adam/tubelet_embedding_33/conv3d_100/bias/v/Read/ReadVariableOpAAdam/tubelet_embedding_33/conv3d_101/kernel/v/Read/ReadVariableOp?Adam/tubelet_embedding_33/conv3d_101/bias/v/Read/ReadVariableOpEAdam/positional_encoder_16/embedding/embeddings/v/Read/ReadVariableOp>Adam/multi_head_attention_6/query/kernel/v/Read/ReadVariableOp<Adam/multi_head_attention_6/query/bias/v/Read/ReadVariableOp<Adam/multi_head_attention_6/key/kernel/v/Read/ReadVariableOp:Adam/multi_head_attention_6/key/bias/v/Read/ReadVariableOp>Adam/multi_head_attention_6/value/kernel/v/Read/ReadVariableOp<Adam/multi_head_attention_6/value/bias/v/Read/ReadVariableOpIAdam/multi_head_attention_6/attention_output/kernel/v/Read/ReadVariableOpGAdam/multi_head_attention_6/attention_output/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOpConst_1*s
Tinl
j2h	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *(
f#R!
__inference__traced_save_497047
й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization_18/gammalayer_normalization_18/betalayer_normalization_19/gammalayer_normalization_19/betalayer_normalization_20/gammalayer_normalization_20/betadense_13/kerneldense_13/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%tubelet_embedding_32/conv3d_96/kernel#tubelet_embedding_32/conv3d_96/bias%tubelet_embedding_32/conv3d_97/kernel#tubelet_embedding_32/conv3d_97/bias%tubelet_embedding_32/conv3d_98/kernel#tubelet_embedding_32/conv3d_98/bias%tubelet_embedding_33/conv3d_99/kernel#tubelet_embedding_33/conv3d_99/bias&tubelet_embedding_33/conv3d_100/kernel$tubelet_embedding_33/conv3d_100/bias&tubelet_embedding_33/conv3d_101/kernel$tubelet_embedding_33/conv3d_101/bias*positional_encoder_16/embedding/embeddings#multi_head_attention_6/query/kernel!multi_head_attention_6/query/bias!multi_head_attention_6/key/kernelmulti_head_attention_6/key/bias#multi_head_attention_6/value/kernel!multi_head_attention_6/value/bias.multi_head_attention_6/attention_output/kernel,multi_head_attention_6/attention_output/biasdense_12/kerneldense_12/biastotalcounttotal_1count_1#Adam/layer_normalization_18/gamma/m"Adam/layer_normalization_18/beta/m#Adam/layer_normalization_19/gamma/m"Adam/layer_normalization_19/beta/m#Adam/layer_normalization_20/gamma/m"Adam/layer_normalization_20/beta/mAdam/dense_13/kernel/mAdam/dense_13/bias/m,Adam/tubelet_embedding_32/conv3d_96/kernel/m*Adam/tubelet_embedding_32/conv3d_96/bias/m,Adam/tubelet_embedding_32/conv3d_97/kernel/m*Adam/tubelet_embedding_32/conv3d_97/bias/m,Adam/tubelet_embedding_32/conv3d_98/kernel/m*Adam/tubelet_embedding_32/conv3d_98/bias/m,Adam/tubelet_embedding_33/conv3d_99/kernel/m*Adam/tubelet_embedding_33/conv3d_99/bias/m-Adam/tubelet_embedding_33/conv3d_100/kernel/m+Adam/tubelet_embedding_33/conv3d_100/bias/m-Adam/tubelet_embedding_33/conv3d_101/kernel/m+Adam/tubelet_embedding_33/conv3d_101/bias/m1Adam/positional_encoder_16/embedding/embeddings/m*Adam/multi_head_attention_6/query/kernel/m(Adam/multi_head_attention_6/query/bias/m(Adam/multi_head_attention_6/key/kernel/m&Adam/multi_head_attention_6/key/bias/m*Adam/multi_head_attention_6/value/kernel/m(Adam/multi_head_attention_6/value/bias/m5Adam/multi_head_attention_6/attention_output/kernel/m3Adam/multi_head_attention_6/attention_output/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/m#Adam/layer_normalization_18/gamma/v"Adam/layer_normalization_18/beta/v#Adam/layer_normalization_19/gamma/v"Adam/layer_normalization_19/beta/v#Adam/layer_normalization_20/gamma/v"Adam/layer_normalization_20/beta/vAdam/dense_13/kernel/vAdam/dense_13/bias/v,Adam/tubelet_embedding_32/conv3d_96/kernel/v*Adam/tubelet_embedding_32/conv3d_96/bias/v,Adam/tubelet_embedding_32/conv3d_97/kernel/v*Adam/tubelet_embedding_32/conv3d_97/bias/v,Adam/tubelet_embedding_32/conv3d_98/kernel/v*Adam/tubelet_embedding_32/conv3d_98/bias/v,Adam/tubelet_embedding_33/conv3d_99/kernel/v*Adam/tubelet_embedding_33/conv3d_99/bias/v-Adam/tubelet_embedding_33/conv3d_100/kernel/v+Adam/tubelet_embedding_33/conv3d_100/bias/v-Adam/tubelet_embedding_33/conv3d_101/kernel/v+Adam/tubelet_embedding_33/conv3d_101/bias/v1Adam/positional_encoder_16/embedding/embeddings/v*Adam/multi_head_attention_6/query/kernel/v(Adam/multi_head_attention_6/query/bias/v(Adam/multi_head_attention_6/key/kernel/v&Adam/multi_head_attention_6/key/bias/v*Adam/multi_head_attention_6/value/kernel/v(Adam/multi_head_attention_6/value/bias/v5Adam/multi_head_attention_6/attention_output/kernel/v3Adam/multi_head_attention_6/attention_output/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/v*r
Tink
i2g*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *+
f&R$
"__inference__traced_restore_497363Жж
Џ
h
L__inference_max_pooling3d_65_layer_call_and_return_conditional_losses_494202

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Џ
h
L__inference_max_pooling3d_66_layer_call_and_return_conditional_losses_494226

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
б
ї
$__inference_signature_wrapper_496102
input_17%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@А
	unknown_4:	А'
	unknown_5: 
	unknown_6: '
	unknown_7: @
	unknown_8:@(
	unknown_9:@А

unknown_10:	А

unknown_11

unknown_12:	(А

unknown_13:	А

unknown_14:	А"

unknown_15:АА

unknown_16:	А"

unknown_17:АА

unknown_18:	А"

unknown_19:АА

unknown_20:	А"

unknown_21:АА

unknown_22:	А

unknown_23:	А

unknown_24:	А

unknown_25:
АА

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8В **
f%R#
!__inference__wrapped_model_494181o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
3
_output_shapes!
:€€€€€€€€€

"
_user_specified_name
input_17: 

_output_shapes
:(
г[
э
C__inference_model_6_layer_call_and_return_conditional_losses_495136

inputs9
tubelet_embedding_32_495057: )
tubelet_embedding_32_495059: 9
tubelet_embedding_32_495061: @)
tubelet_embedding_32_495063:@:
tubelet_embedding_32_495065:@А*
tubelet_embedding_32_495067:	А9
tubelet_embedding_33_495070: )
tubelet_embedding_33_495072: 9
tubelet_embedding_33_495074: @)
tubelet_embedding_33_495076:@:
tubelet_embedding_33_495078:@А*
tubelet_embedding_33_495080:	А 
positional_encoder_16_495084/
positional_encoder_16_495086:	(А,
layer_normalization_18_495089:	А,
layer_normalization_18_495091:	А5
multi_head_attention_6_495094:АА0
multi_head_attention_6_495096:	А5
multi_head_attention_6_495098:АА0
multi_head_attention_6_495100:	А5
multi_head_attention_6_495102:АА0
multi_head_attention_6_495104:	А5
multi_head_attention_6_495106:АА,
multi_head_attention_6_495108:	А,
layer_normalization_19_495113:	А,
layer_normalization_19_495115:	А'
sequential_6_495118:
АА"
sequential_6_495120:	А,
layer_normalization_20_495124:	А,
layer_normalization_20_495126:	А"
dense_13_495130:	А
dense_13_495132:
identityИҐ dense_13/StatefulPartitionedCallҐ.layer_normalization_18/StatefulPartitionedCallҐ.layer_normalization_19/StatefulPartitionedCallҐ.layer_normalization_20/StatefulPartitionedCallҐ.multi_head_attention_6/StatefulPartitionedCallҐ-positional_encoder_16/StatefulPartitionedCallҐ$sequential_6/StatefulPartitionedCallҐ,tubelet_embedding_32/StatefulPartitionedCallҐ,tubelet_embedding_33/StatefulPartitionedCallМ
/tf.__operators__.getitem_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               а
)tf.__operators__.getitem_37/strided_sliceStridedSliceinputs8tf.__operators__.getitem_37/strided_slice/stack:output:0:tf.__operators__.getitem_37/strided_slice/stack_1:output:0:tf.__operators__.getitem_37/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_maskМ
/tf.__operators__.getitem_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               а
)tf.__operators__.getitem_36/strided_sliceStridedSliceinputs8tf.__operators__.getitem_36/strided_slice/stack:output:0:tf.__operators__.getitem_36/strided_slice/stack_1:output:0:tf.__operators__.getitem_36/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_mask“
,tubelet_embedding_32/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_36/strided_slice:output:0tubelet_embedding_32_495057tubelet_embedding_32_495059tubelet_embedding_32_495061tubelet_embedding_32_495063tubelet_embedding_32_495065tubelet_embedding_32_495067*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_tubelet_embedding_32_layer_call_and_return_conditional_losses_494440“
,tubelet_embedding_33/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_37/strided_slice:output:0tubelet_embedding_33_495070tubelet_embedding_33_495072tubelet_embedding_33_495074tubelet_embedding_33_495076tubelet_embedding_33_495078tubelet_embedding_33_495080*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_tubelet_embedding_33_layer_call_and_return_conditional_losses_494490µ
concatenate_13/PartitionedCallPartitionedCall5tubelet_embedding_32/StatefulPartitionedCall:output:05tubelet_embedding_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *S
fNRL
J__inference_concatenate_13_layer_call_and_return_conditional_losses_494511ќ
-positional_encoder_16/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0positional_encoder_16_495084positional_encoder_16_495086*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Z
fURS
Q__inference_positional_encoder_16_layer_call_and_return_conditional_losses_494525в
.layer_normalization_18/StatefulPartitionedCallStatefulPartitionedCall6positional_encoder_16/StatefulPartitionedCall:output:0layer_normalization_18_495089layer_normalization_18_495091*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_18_layer_call_and_return_conditional_losses_494553€
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_18/StatefulPartitionedCall:output:07layer_normalization_18/StatefulPartitionedCall:output:0multi_head_attention_6_495094multi_head_attention_6_495096multi_head_attention_6_495098multi_head_attention_6_495100multi_head_attention_6_495102multi_head_attention_6_495104multi_head_attention_6_495106multi_head_attention_6_495108*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:€€€€€€€€€(А:€€€€€€€€€((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_494891®
add_12/PartitionedCallPartitionedCall7multi_head_attention_6/StatefulPartitionedCall:output:06positional_encoder_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_494620Ћ
.layer_normalization_19/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0layer_normalization_19_495113layer_normalization_19_495115*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_19_layer_call_and_return_conditional_losses_494644ї
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_19/StatefulPartitionedCall:output:0sequential_6_495118sequential_6_495120*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494342З
add_13/PartitionedCallPartitionedCall-sequential_6/StatefulPartitionedCall:output:0add_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *K
fFRD
B__inference_add_13_layer_call_and_return_conditional_losses_494661Ћ
.layer_normalization_20/StatefulPartitionedCallStatefulPartitionedCalladd_13/PartitionedCall:output:0layer_normalization_20_495124layer_normalization_20_495126*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_20_layer_call_and_return_conditional_losses_494685У
*global_average_pooling1d_6/PartitionedCallPartitionedCall7layer_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *_
fZRX
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_494386Ґ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_6/PartitionedCall:output:0dense_13_495130dense_13_495132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_494702x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€в
NoOpNoOp!^dense_13/StatefulPartitionedCall/^layer_normalization_18/StatefulPartitionedCall/^layer_normalization_19/StatefulPartitionedCall/^layer_normalization_20/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall.^positional_encoder_16/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall-^tubelet_embedding_32/StatefulPartitionedCall-^tubelet_embedding_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2`
.layer_normalization_18/StatefulPartitionedCall.layer_normalization_18/StatefulPartitionedCall2`
.layer_normalization_19/StatefulPartitionedCall.layer_normalization_19/StatefulPartitionedCall2`
.layer_normalization_20/StatefulPartitionedCall.layer_normalization_20/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2^
-positional_encoder_16/StatefulPartitionedCall-positional_encoder_16/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2\
,tubelet_embedding_32/StatefulPartitionedCall,tubelet_embedding_32/StatefulPartitionedCall2\
,tubelet_embedding_33/StatefulPartitionedCall,tubelet_embedding_33/StatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_nameinputs: 

_output_shapes
:(
Ж-
√
P__inference_tubelet_embedding_32_layer_call_and_return_conditional_losses_494440

videosF
(conv3d_96_conv3d_readvariableop_resource: 7
)conv3d_96_biasadd_readvariableop_resource: F
(conv3d_97_conv3d_readvariableop_resource: @7
)conv3d_97_biasadd_readvariableop_resource:@G
(conv3d_98_conv3d_readvariableop_resource:@А8
)conv3d_98_biasadd_readvariableop_resource:	А
identityИҐ conv3d_96/BiasAdd/ReadVariableOpҐconv3d_96/Conv3D/ReadVariableOpҐ conv3d_97/BiasAdd/ReadVariableOpҐconv3d_97/Conv3D/ReadVariableOpҐ conv3d_98/BiasAdd/ReadVariableOpҐconv3d_98/Conv3D/ReadVariableOpФ
conv3d_96/Conv3D/ReadVariableOpReadVariableOp(conv3d_96_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0≤
conv3d_96/Conv3DConv3Dvideos'conv3d_96/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
paddingSAME*
strides	
Ж
 conv3d_96/BiasAdd/ReadVariableOpReadVariableOp)conv3d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Я
conv3d_96/BiasAddBiasAddconv3d_96/Conv3D:output:0(conv3d_96/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 p
conv3d_96/ReluReluconv3d_96/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
 Ѕ
max_pooling3d_64/MaxPool3D	MaxPool3Dconv3d_96/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
ksize	
*
paddingVALID*
strides	
Ф
conv3d_97/Conv3D/ReadVariableOpReadVariableOp(conv3d_97_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0ѕ
conv3d_97/Conv3DConv3D#max_pooling3d_64/MaxPool3D:output:0'conv3d_97/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
paddingSAME*
strides	
Ж
 conv3d_97/BiasAdd/ReadVariableOpReadVariableOp)conv3d_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Я
conv3d_97/BiasAddBiasAddconv3d_97/Conv3D:output:0(conv3d_97/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@p
conv3d_97/ReluReluconv3d_97/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
@Ѕ
max_pooling3d_65/MaxPool3D	MaxPool3Dconv3d_97/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
ksize	
*
paddingVALID*
strides	
Х
conv3d_98/Conv3D/ReadVariableOpReadVariableOp(conv3d_98_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0–
conv3d_98/Conv3DConv3D#max_pooling3d_65/MaxPool3D:output:0'conv3d_98/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
paddingSAME*
strides	
З
 conv3d_98/BiasAdd/ReadVariableOpReadVariableOp)conv3d_98_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0†
conv3d_98/BiasAddBiasAddconv3d_98/Conv3D:output:0(conv3d_98/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
Аƒ
average_pooling3d_32/AvgPool3D	AvgPool3Dconv3d_98/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
ksize	
*
paddingVALID*
strides	
g
reshape_32/ShapeShape'average_pooling3d_32/AvgPool3D:output:0*
T0*
_output_shapes
:h
reshape_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
reshape_32/strided_sliceStridedSlicereshape_32/Shape:output:0'reshape_32/strided_slice/stack:output:0)reshape_32/strided_slice/stack_1:output:0)reshape_32/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
reshape_32/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€]
reshape_32/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Аї
reshape_32/Reshape/shapePack!reshape_32/strided_slice:output:0#reshape_32/Reshape/shape/1:output:0#reshape_32/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:†
reshape_32/ReshapeReshape'average_pooling3d_32/AvgPool3D:output:0!reshape_32/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аo
IdentityIdentityreshape_32/Reshape:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АХ
NoOpNoOp!^conv3d_96/BiasAdd/ReadVariableOp ^conv3d_96/Conv3D/ReadVariableOp!^conv3d_97/BiasAdd/ReadVariableOp ^conv3d_97/Conv3D/ReadVariableOp!^conv3d_98/BiasAdd/ReadVariableOp ^conv3d_98/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€
: : : : : : 2D
 conv3d_96/BiasAdd/ReadVariableOp conv3d_96/BiasAdd/ReadVariableOp2B
conv3d_96/Conv3D/ReadVariableOpconv3d_96/Conv3D/ReadVariableOp2D
 conv3d_97/BiasAdd/ReadVariableOp conv3d_97/BiasAdd/ReadVariableOp2B
conv3d_97/Conv3D/ReadVariableOpconv3d_97/Conv3D/ReadVariableOp2D
 conv3d_98/BiasAdd/ReadVariableOp conv3d_98/BiasAdd/ReadVariableOp2B
conv3d_98/Conv3D/ReadVariableOpconv3d_98/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_namevideos
н+
Ч
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_496357	
query	
valueC
+query_einsum_einsum_readvariableop_resource:АА4
!query_add_readvariableop_resource:	АA
)key_einsum_einsum_readvariableop_resource:АА2
key_add_readvariableop_resource:	АC
+value_einsum_einsum_readvariableop_resource:АА4
!value_add_readvariableop_resource:	АN
6attention_output_einsum_einsum_readvariableop_resource:АА;
,attention_output_add_readvariableop_resource:	А
identity

identity_1ИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpФ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde{
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АР
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abdew
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes
:	А*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АФ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde{
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€(АП
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€((*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€((q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€((¶
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationacbe,aecd->abcd™
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А*
equationabcd,cde->abeН
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аr

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€((Ў
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€(А:€€€€€€€€€(А: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namevalue
х
Ґ
7__inference_layer_normalization_18_layer_call_fn_496251

inputs
unknown:	А
	unknown_0:	А
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_18_layer_call_and_return_conditional_losses_494553t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
х
Ґ
7__inference_layer_normalization_20_layer_call_fn_496558

inputs
unknown:	А
	unknown_0:	А
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_20_layer_call_and_return_conditional_losses_494685t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
ю
•
-__inference_sequential_6_layer_call_fn_494358
dense_12_input
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCalldense_12_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494342t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:€€€€€€€€€(А
(
_user_specified_namedense_12_input
з
„
H__inference_sequential_6_layer_call_and_return_conditional_losses_494367
dense_12_input#
dense_12_494361:
АА
dense_12_494363:	А
identityИҐ dense_12/StatefulPartitionedCallВ
 dense_12/StatefulPartitionedCallStatefulPartitionedCalldense_12_inputdense_12_494361dense_12_494363*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_494298}
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аi
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:\ X
,
_output_shapes
:€€€€€€€€€(А
(
_user_specified_namedense_12_input
®І
гM
"__inference__traced_restore_497363
file_prefix<
-assignvariableop_layer_normalization_18_gamma:	А=
.assignvariableop_1_layer_normalization_18_beta:	А>
/assignvariableop_2_layer_normalization_19_gamma:	А=
.assignvariableop_3_layer_normalization_19_beta:	А>
/assignvariableop_4_layer_normalization_20_gamma:	А=
.assignvariableop_5_layer_normalization_20_beta:	А5
"assignvariableop_6_dense_13_kernel:	А.
 assignvariableop_7_dense_13_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: W
9assignvariableop_13_tubelet_embedding_32_conv3d_96_kernel: E
7assignvariableop_14_tubelet_embedding_32_conv3d_96_bias: W
9assignvariableop_15_tubelet_embedding_32_conv3d_97_kernel: @E
7assignvariableop_16_tubelet_embedding_32_conv3d_97_bias:@X
9assignvariableop_17_tubelet_embedding_32_conv3d_98_kernel:@АF
7assignvariableop_18_tubelet_embedding_32_conv3d_98_bias:	АW
9assignvariableop_19_tubelet_embedding_33_conv3d_99_kernel: E
7assignvariableop_20_tubelet_embedding_33_conv3d_99_bias: X
:assignvariableop_21_tubelet_embedding_33_conv3d_100_kernel: @F
8assignvariableop_22_tubelet_embedding_33_conv3d_100_bias:@Y
:assignvariableop_23_tubelet_embedding_33_conv3d_101_kernel:@АG
8assignvariableop_24_tubelet_embedding_33_conv3d_101_bias:	АQ
>assignvariableop_25_positional_encoder_16_embedding_embeddings:	(АO
7assignvariableop_26_multi_head_attention_6_query_kernel:ААH
5assignvariableop_27_multi_head_attention_6_query_bias:	АM
5assignvariableop_28_multi_head_attention_6_key_kernel:ААF
3assignvariableop_29_multi_head_attention_6_key_bias:	АO
7assignvariableop_30_multi_head_attention_6_value_kernel:ААH
5assignvariableop_31_multi_head_attention_6_value_bias:	АZ
Bassignvariableop_32_multi_head_attention_6_attention_output_kernel:ААO
@assignvariableop_33_multi_head_attention_6_attention_output_bias:	А7
#assignvariableop_34_dense_12_kernel:
АА0
!assignvariableop_35_dense_12_bias:	А#
assignvariableop_36_total: #
assignvariableop_37_count: %
assignvariableop_38_total_1: %
assignvariableop_39_count_1: F
7assignvariableop_40_adam_layer_normalization_18_gamma_m:	АE
6assignvariableop_41_adam_layer_normalization_18_beta_m:	АF
7assignvariableop_42_adam_layer_normalization_19_gamma_m:	АE
6assignvariableop_43_adam_layer_normalization_19_beta_m:	АF
7assignvariableop_44_adam_layer_normalization_20_gamma_m:	АE
6assignvariableop_45_adam_layer_normalization_20_beta_m:	А=
*assignvariableop_46_adam_dense_13_kernel_m:	А6
(assignvariableop_47_adam_dense_13_bias_m:^
@assignvariableop_48_adam_tubelet_embedding_32_conv3d_96_kernel_m: L
>assignvariableop_49_adam_tubelet_embedding_32_conv3d_96_bias_m: ^
@assignvariableop_50_adam_tubelet_embedding_32_conv3d_97_kernel_m: @L
>assignvariableop_51_adam_tubelet_embedding_32_conv3d_97_bias_m:@_
@assignvariableop_52_adam_tubelet_embedding_32_conv3d_98_kernel_m:@АM
>assignvariableop_53_adam_tubelet_embedding_32_conv3d_98_bias_m:	А^
@assignvariableop_54_adam_tubelet_embedding_33_conv3d_99_kernel_m: L
>assignvariableop_55_adam_tubelet_embedding_33_conv3d_99_bias_m: _
Aassignvariableop_56_adam_tubelet_embedding_33_conv3d_100_kernel_m: @M
?assignvariableop_57_adam_tubelet_embedding_33_conv3d_100_bias_m:@`
Aassignvariableop_58_adam_tubelet_embedding_33_conv3d_101_kernel_m:@АN
?assignvariableop_59_adam_tubelet_embedding_33_conv3d_101_bias_m:	АX
Eassignvariableop_60_adam_positional_encoder_16_embedding_embeddings_m:	(АV
>assignvariableop_61_adam_multi_head_attention_6_query_kernel_m:ААO
<assignvariableop_62_adam_multi_head_attention_6_query_bias_m:	АT
<assignvariableop_63_adam_multi_head_attention_6_key_kernel_m:ААM
:assignvariableop_64_adam_multi_head_attention_6_key_bias_m:	АV
>assignvariableop_65_adam_multi_head_attention_6_value_kernel_m:ААO
<assignvariableop_66_adam_multi_head_attention_6_value_bias_m:	Аa
Iassignvariableop_67_adam_multi_head_attention_6_attention_output_kernel_m:ААV
Gassignvariableop_68_adam_multi_head_attention_6_attention_output_bias_m:	А>
*assignvariableop_69_adam_dense_12_kernel_m:
АА7
(assignvariableop_70_adam_dense_12_bias_m:	АF
7assignvariableop_71_adam_layer_normalization_18_gamma_v:	АE
6assignvariableop_72_adam_layer_normalization_18_beta_v:	АF
7assignvariableop_73_adam_layer_normalization_19_gamma_v:	АE
6assignvariableop_74_adam_layer_normalization_19_beta_v:	АF
7assignvariableop_75_adam_layer_normalization_20_gamma_v:	АE
6assignvariableop_76_adam_layer_normalization_20_beta_v:	А=
*assignvariableop_77_adam_dense_13_kernel_v:	А6
(assignvariableop_78_adam_dense_13_bias_v:^
@assignvariableop_79_adam_tubelet_embedding_32_conv3d_96_kernel_v: L
>assignvariableop_80_adam_tubelet_embedding_32_conv3d_96_bias_v: ^
@assignvariableop_81_adam_tubelet_embedding_32_conv3d_97_kernel_v: @L
>assignvariableop_82_adam_tubelet_embedding_32_conv3d_97_bias_v:@_
@assignvariableop_83_adam_tubelet_embedding_32_conv3d_98_kernel_v:@АM
>assignvariableop_84_adam_tubelet_embedding_32_conv3d_98_bias_v:	А^
@assignvariableop_85_adam_tubelet_embedding_33_conv3d_99_kernel_v: L
>assignvariableop_86_adam_tubelet_embedding_33_conv3d_99_bias_v: _
Aassignvariableop_87_adam_tubelet_embedding_33_conv3d_100_kernel_v: @M
?assignvariableop_88_adam_tubelet_embedding_33_conv3d_100_bias_v:@`
Aassignvariableop_89_adam_tubelet_embedding_33_conv3d_101_kernel_v:@АN
?assignvariableop_90_adam_tubelet_embedding_33_conv3d_101_bias_v:	АX
Eassignvariableop_91_adam_positional_encoder_16_embedding_embeddings_v:	(АV
>assignvariableop_92_adam_multi_head_attention_6_query_kernel_v:ААO
<assignvariableop_93_adam_multi_head_attention_6_query_bias_v:	АT
<assignvariableop_94_adam_multi_head_attention_6_key_kernel_v:ААM
:assignvariableop_95_adam_multi_head_attention_6_key_bias_v:	АV
>assignvariableop_96_adam_multi_head_attention_6_value_kernel_v:ААO
<assignvariableop_97_adam_multi_head_attention_6_value_bias_v:	Аa
Iassignvariableop_98_adam_multi_head_attention_6_attention_output_kernel_v:ААV
Gassignvariableop_99_adam_multi_head_attention_6_attention_output_bias_v:	А?
+assignvariableop_100_adam_dense_12_kernel_v:
АА8
)assignvariableop_101_adam_dense_12_bias_v:	А
identity_103ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_100ҐAssignVariableOp_101ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_89ҐAssignVariableOp_9ҐAssignVariableOp_90ҐAssignVariableOp_91ҐAssignVariableOp_92ҐAssignVariableOp_93ҐAssignVariableOp_94ҐAssignVariableOp_95ҐAssignVariableOp_96ҐAssignVariableOp_97ҐAssignVariableOp_98ҐAssignVariableOp_99†2
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:g*
dtype0*∆1
valueЉ1Bє1gB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЅ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:g*
dtype0*г
valueўB÷gB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B §
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*≤
_output_shapesЯ
Ь:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*u
dtypesk
i2g	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOpAssignVariableOp-assignvariableop_layer_normalization_18_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_1AssignVariableOp.assignvariableop_1_layer_normalization_18_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp/assignvariableop_2_layer_normalization_19_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_3AssignVariableOp.assignvariableop_3_layer_normalization_19_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_4AssignVariableOp/assignvariableop_4_layer_normalization_20_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_5AssignVariableOp.assignvariableop_5_layer_normalization_20_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_13_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_13AssignVariableOp9assignvariableop_13_tubelet_embedding_32_conv3d_96_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_14AssignVariableOp7assignvariableop_14_tubelet_embedding_32_conv3d_96_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_15AssignVariableOp9assignvariableop_15_tubelet_embedding_32_conv3d_97_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_16AssignVariableOp7assignvariableop_16_tubelet_embedding_32_conv3d_97_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_17AssignVariableOp9assignvariableop_17_tubelet_embedding_32_conv3d_98_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_18AssignVariableOp7assignvariableop_18_tubelet_embedding_32_conv3d_98_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_19AssignVariableOp9assignvariableop_19_tubelet_embedding_33_conv3d_99_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_20AssignVariableOp7assignvariableop_20_tubelet_embedding_33_conv3d_99_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_21AssignVariableOp:assignvariableop_21_tubelet_embedding_33_conv3d_100_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_22AssignVariableOp8assignvariableop_22_tubelet_embedding_33_conv3d_100_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_23AssignVariableOp:assignvariableop_23_tubelet_embedding_33_conv3d_101_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_24AssignVariableOp8assignvariableop_24_tubelet_embedding_33_conv3d_101_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_25AssignVariableOp>assignvariableop_25_positional_encoder_16_embedding_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_26AssignVariableOp7assignvariableop_26_multi_head_attention_6_query_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_27AssignVariableOp5assignvariableop_27_multi_head_attention_6_query_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_28AssignVariableOp5assignvariableop_28_multi_head_attention_6_key_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_29AssignVariableOp3assignvariableop_29_multi_head_attention_6_key_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_30AssignVariableOp7assignvariableop_30_multi_head_attention_6_value_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_31AssignVariableOp5assignvariableop_31_multi_head_attention_6_value_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_32AssignVariableOpBassignvariableop_32_multi_head_attention_6_attention_output_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_33AssignVariableOp@assignvariableop_33_multi_head_attention_6_attention_output_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_12_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_12_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_36AssignVariableOpassignvariableop_36_totalIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_37AssignVariableOpassignvariableop_37_countIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_39AssignVariableOpassignvariableop_39_count_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_layer_normalization_18_gamma_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_layer_normalization_18_beta_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_42AssignVariableOp7assignvariableop_42_adam_layer_normalization_19_gamma_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_layer_normalization_19_beta_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adam_layer_normalization_20_gamma_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_layer_normalization_20_beta_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dense_13_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense_13_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_48AssignVariableOp@assignvariableop_48_adam_tubelet_embedding_32_conv3d_96_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_49AssignVariableOp>assignvariableop_49_adam_tubelet_embedding_32_conv3d_96_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_50AssignVariableOp@assignvariableop_50_adam_tubelet_embedding_32_conv3d_97_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_51AssignVariableOp>assignvariableop_51_adam_tubelet_embedding_32_conv3d_97_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_52AssignVariableOp@assignvariableop_52_adam_tubelet_embedding_32_conv3d_98_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_53AssignVariableOp>assignvariableop_53_adam_tubelet_embedding_32_conv3d_98_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_54AssignVariableOp@assignvariableop_54_adam_tubelet_embedding_33_conv3d_99_kernel_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_55AssignVariableOp>assignvariableop_55_adam_tubelet_embedding_33_conv3d_99_bias_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_56AssignVariableOpAassignvariableop_56_adam_tubelet_embedding_33_conv3d_100_kernel_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:∞
AssignVariableOp_57AssignVariableOp?assignvariableop_57_adam_tubelet_embedding_33_conv3d_100_bias_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_58AssignVariableOpAassignvariableop_58_adam_tubelet_embedding_33_conv3d_101_kernel_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:∞
AssignVariableOp_59AssignVariableOp?assignvariableop_59_adam_tubelet_embedding_33_conv3d_101_bias_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_60AssignVariableOpEassignvariableop_60_adam_positional_encoder_16_embedding_embeddings_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_61AssignVariableOp>assignvariableop_61_adam_multi_head_attention_6_query_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_62AssignVariableOp<assignvariableop_62_adam_multi_head_attention_6_query_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_63AssignVariableOp<assignvariableop_63_adam_multi_head_attention_6_key_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_64AssignVariableOp:assignvariableop_64_adam_multi_head_attention_6_key_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_65AssignVariableOp>assignvariableop_65_adam_multi_head_attention_6_value_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_66AssignVariableOp<assignvariableop_66_adam_multi_head_attention_6_value_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_67AssignVariableOpIassignvariableop_67_adam_multi_head_attention_6_attention_output_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_68AssignVariableOpGassignvariableop_68_adam_multi_head_attention_6_attention_output_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dense_12_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_dense_12_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_layer_normalization_18_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_layer_normalization_18_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_73AssignVariableOp7assignvariableop_73_adam_layer_normalization_19_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_layer_normalization_19_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_layer_normalization_20_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_layer_normalization_20_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_13_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_13_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_79AssignVariableOp@assignvariableop_79_adam_tubelet_embedding_32_conv3d_96_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_80AssignVariableOp>assignvariableop_80_adam_tubelet_embedding_32_conv3d_96_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_81AssignVariableOp@assignvariableop_81_adam_tubelet_embedding_32_conv3d_97_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_82AssignVariableOp>assignvariableop_82_adam_tubelet_embedding_32_conv3d_97_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_83AssignVariableOp@assignvariableop_83_adam_tubelet_embedding_32_conv3d_98_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_84AssignVariableOp>assignvariableop_84_adam_tubelet_embedding_32_conv3d_98_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_85AssignVariableOp@assignvariableop_85_adam_tubelet_embedding_33_conv3d_99_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_86AssignVariableOp>assignvariableop_86_adam_tubelet_embedding_33_conv3d_99_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_87AssignVariableOpAassignvariableop_87_adam_tubelet_embedding_33_conv3d_100_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:∞
AssignVariableOp_88AssignVariableOp?assignvariableop_88_adam_tubelet_embedding_33_conv3d_100_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_89AssignVariableOpAassignvariableop_89_adam_tubelet_embedding_33_conv3d_101_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:∞
AssignVariableOp_90AssignVariableOp?assignvariableop_90_adam_tubelet_embedding_33_conv3d_101_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_91AssignVariableOpEassignvariableop_91_adam_positional_encoder_16_embedding_embeddings_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_92AssignVariableOp>assignvariableop_92_adam_multi_head_attention_6_query_kernel_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_93AssignVariableOp<assignvariableop_93_adam_multi_head_attention_6_query_bias_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_94AssignVariableOp<assignvariableop_94_adam_multi_head_attention_6_key_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_95AssignVariableOp:assignvariableop_95_adam_multi_head_attention_6_key_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_96AssignVariableOp>assignvariableop_96_adam_multi_head_attention_6_value_kernel_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_97AssignVariableOp<assignvariableop_97_adam_multi_head_attention_6_value_bias_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_98AssignVariableOpIassignvariableop_98_adam_multi_head_attention_6_attention_output_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_99AssignVariableOpGassignvariableop_99_adam_multi_head_attention_6_attention_output_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_100AssignVariableOp+assignvariableop_100_adam_dense_12_kernel_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_101AssignVariableOp)assignvariableop_101_adam_dense_12_bias_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ц
Identity_102Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_103IdentityIdentity_102:output:0^NoOp_1*
T0*
_output_shapes
: В
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_103Identity_103:output:0*г
_input_shapes—
ќ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ю
У
R__inference_layer_normalization_18_layer_call_and_return_conditional_losses_494553

inputs4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(М
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ж
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0В
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АА
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
®р
£%
!__inference__wrapped_model_494181
input_17c
Emodel_6_tubelet_embedding_32_conv3d_96_conv3d_readvariableop_resource: T
Fmodel_6_tubelet_embedding_32_conv3d_96_biasadd_readvariableop_resource: c
Emodel_6_tubelet_embedding_32_conv3d_97_conv3d_readvariableop_resource: @T
Fmodel_6_tubelet_embedding_32_conv3d_97_biasadd_readvariableop_resource:@d
Emodel_6_tubelet_embedding_32_conv3d_98_conv3d_readvariableop_resource:@АU
Fmodel_6_tubelet_embedding_32_conv3d_98_biasadd_readvariableop_resource:	Аc
Emodel_6_tubelet_embedding_33_conv3d_99_conv3d_readvariableop_resource: T
Fmodel_6_tubelet_embedding_33_conv3d_99_biasadd_readvariableop_resource: d
Fmodel_6_tubelet_embedding_33_conv3d_100_conv3d_readvariableop_resource: @U
Gmodel_6_tubelet_embedding_33_conv3d_100_biasadd_readvariableop_resource:@e
Fmodel_6_tubelet_embedding_33_conv3d_101_conv3d_readvariableop_resource:@АV
Gmodel_6_tubelet_embedding_33_conv3d_101_biasadd_readvariableop_resource:	А(
$model_6_positional_encoder_16_494044R
?model_6_positional_encoder_16_embedding_embedding_lookup_494046:	(АS
Dmodel_6_layer_normalization_18_batchnorm_mul_readvariableop_resource:	АO
@model_6_layer_normalization_18_batchnorm_readvariableop_resource:	Аb
Jmodel_6_multi_head_attention_6_query_einsum_einsum_readvariableop_resource:ААS
@model_6_multi_head_attention_6_query_add_readvariableop_resource:	А`
Hmodel_6_multi_head_attention_6_key_einsum_einsum_readvariableop_resource:ААQ
>model_6_multi_head_attention_6_key_add_readvariableop_resource:	Аb
Jmodel_6_multi_head_attention_6_value_einsum_einsum_readvariableop_resource:ААS
@model_6_multi_head_attention_6_value_add_readvariableop_resource:	Аm
Umodel_6_multi_head_attention_6_attention_output_einsum_einsum_readvariableop_resource:ААZ
Kmodel_6_multi_head_attention_6_attention_output_add_readvariableop_resource:	АS
Dmodel_6_layer_normalization_19_batchnorm_mul_readvariableop_resource:	АO
@model_6_layer_normalization_19_batchnorm_readvariableop_resource:	АS
?model_6_sequential_6_dense_12_tensordot_readvariableop_resource:
ААL
=model_6_sequential_6_dense_12_biasadd_readvariableop_resource:	АS
Dmodel_6_layer_normalization_20_batchnorm_mul_readvariableop_resource:	АO
@model_6_layer_normalization_20_batchnorm_readvariableop_resource:	АB
/model_6_dense_13_matmul_readvariableop_resource:	А>
0model_6_dense_13_biasadd_readvariableop_resource:
identityИҐ'model_6/dense_13/BiasAdd/ReadVariableOpҐ&model_6/dense_13/MatMul/ReadVariableOpҐ7model_6/layer_normalization_18/batchnorm/ReadVariableOpҐ;model_6/layer_normalization_18/batchnorm/mul/ReadVariableOpҐ7model_6/layer_normalization_19/batchnorm/ReadVariableOpҐ;model_6/layer_normalization_19/batchnorm/mul/ReadVariableOpҐ7model_6/layer_normalization_20/batchnorm/ReadVariableOpҐ;model_6/layer_normalization_20/batchnorm/mul/ReadVariableOpҐBmodel_6/multi_head_attention_6/attention_output/add/ReadVariableOpҐLmodel_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpҐ5model_6/multi_head_attention_6/key/add/ReadVariableOpҐ?model_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOpҐ7model_6/multi_head_attention_6/query/add/ReadVariableOpҐAmodel_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpҐ7model_6/multi_head_attention_6/value/add/ReadVariableOpҐAmodel_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpҐ8model_6/positional_encoder_16/embedding/embedding_lookupҐ4model_6/sequential_6/dense_12/BiasAdd/ReadVariableOpҐ6model_6/sequential_6/dense_12/Tensordot/ReadVariableOpҐ=model_6/tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOpҐ<model_6/tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOpҐ=model_6/tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOpҐ<model_6/tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOpҐ=model_6/tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOpҐ<model_6/tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOpҐ>model_6/tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOpҐ=model_6/tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOpҐ>model_6/tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOpҐ=model_6/tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOpҐ=model_6/tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOpҐ<model_6/tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOpФ
7model_6/tf.__operators__.getitem_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   Ц
9model_6/tf.__operators__.getitem_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    Ц
9model_6/tf.__operators__.getitem_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               В
1model_6/tf.__operators__.getitem_37/strided_sliceStridedSliceinput_17@model_6/tf.__operators__.getitem_37/strided_slice/stack:output:0Bmodel_6/tf.__operators__.getitem_37/strided_slice/stack_1:output:0Bmodel_6/tf.__operators__.getitem_37/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_maskФ
7model_6/tf.__operators__.getitem_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    Ц
9model_6/tf.__operators__.getitem_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   Ц
9model_6/tf.__operators__.getitem_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               В
1model_6/tf.__operators__.getitem_36/strided_sliceStridedSliceinput_17@model_6/tf.__operators__.getitem_36/strided_slice/stack:output:0Bmodel_6/tf.__operators__.getitem_36/strided_slice/stack_1:output:0Bmodel_6/tf.__operators__.getitem_36/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_maskќ
<model_6/tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOpReadVariableOpEmodel_6_tubelet_embedding_32_conv3d_96_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0†
-model_6/tubelet_embedding_32/conv3d_96/Conv3DConv3D:model_6/tf.__operators__.getitem_36/strided_slice:output:0Dmodel_6/tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
paddingSAME*
strides	
ј
=model_6/tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOpReadVariableOpFmodel_6_tubelet_embedding_32_conv3d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
.model_6/tubelet_embedding_32/conv3d_96/BiasAddBiasAdd6model_6/tubelet_embedding_32/conv3d_96/Conv3D:output:0Emodel_6/tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 ™
+model_6/tubelet_embedding_32/conv3d_96/ReluRelu7model_6/tubelet_embedding_32/conv3d_96/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
 ы
7model_6/tubelet_embedding_32/max_pooling3d_64/MaxPool3D	MaxPool3D9model_6/tubelet_embedding_32/conv3d_96/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
ksize	
*
paddingVALID*
strides	
ќ
<model_6/tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOpReadVariableOpEmodel_6_tubelet_embedding_32_conv3d_97_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0¶
-model_6/tubelet_embedding_32/conv3d_97/Conv3DConv3D@model_6/tubelet_embedding_32/max_pooling3d_64/MaxPool3D:output:0Dmodel_6/tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
paddingSAME*
strides	
ј
=model_6/tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOpReadVariableOpFmodel_6_tubelet_embedding_32_conv3d_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ц
.model_6/tubelet_embedding_32/conv3d_97/BiasAddBiasAdd6model_6/tubelet_embedding_32/conv3d_97/Conv3D:output:0Emodel_6/tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@™
+model_6/tubelet_embedding_32/conv3d_97/ReluRelu7model_6/tubelet_embedding_32/conv3d_97/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
@ы
7model_6/tubelet_embedding_32/max_pooling3d_65/MaxPool3D	MaxPool3D9model_6/tubelet_embedding_32/conv3d_97/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
ksize	
*
paddingVALID*
strides	
ѕ
<model_6/tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOpReadVariableOpEmodel_6_tubelet_embedding_32_conv3d_98_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0І
-model_6/tubelet_embedding_32/conv3d_98/Conv3DConv3D@model_6/tubelet_embedding_32/max_pooling3d_65/MaxPool3D:output:0Dmodel_6/tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
paddingSAME*
strides	
Ѕ
=model_6/tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOpReadVariableOpFmodel_6_tubelet_embedding_32_conv3d_98_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ч
.model_6/tubelet_embedding_32/conv3d_98/BiasAddBiasAdd6model_6/tubelet_embedding_32/conv3d_98/Conv3D:output:0Emodel_6/tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
Аю
;model_6/tubelet_embedding_32/average_pooling3d_32/AvgPool3D	AvgPool3D7model_6/tubelet_embedding_32/conv3d_98/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
ksize	
*
paddingVALID*
strides	
°
-model_6/tubelet_embedding_32/reshape_32/ShapeShapeDmodel_6/tubelet_embedding_32/average_pooling3d_32/AvgPool3D:output:0*
T0*
_output_shapes
:Е
;model_6/tubelet_embedding_32/reshape_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: З
=model_6/tubelet_embedding_32/reshape_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:З
=model_6/tubelet_embedding_32/reshape_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Щ
5model_6/tubelet_embedding_32/reshape_32/strided_sliceStridedSlice6model_6/tubelet_embedding_32/reshape_32/Shape:output:0Dmodel_6/tubelet_embedding_32/reshape_32/strided_slice/stack:output:0Fmodel_6/tubelet_embedding_32/reshape_32/strided_slice/stack_1:output:0Fmodel_6/tubelet_embedding_32/reshape_32/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskВ
7model_6/tubelet_embedding_32/reshape_32/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€z
7model_6/tubelet_embedding_32/reshape_32/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Аѓ
5model_6/tubelet_embedding_32/reshape_32/Reshape/shapePack>model_6/tubelet_embedding_32/reshape_32/strided_slice:output:0@model_6/tubelet_embedding_32/reshape_32/Reshape/shape/1:output:0@model_6/tubelet_embedding_32/reshape_32/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:ч
/model_6/tubelet_embedding_32/reshape_32/ReshapeReshapeDmodel_6/tubelet_embedding_32/average_pooling3d_32/AvgPool3D:output:0>model_6/tubelet_embedding_32/reshape_32/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аќ
<model_6/tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOpReadVariableOpEmodel_6_tubelet_embedding_33_conv3d_99_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0†
-model_6/tubelet_embedding_33/conv3d_99/Conv3DConv3D:model_6/tf.__operators__.getitem_37/strided_slice:output:0Dmodel_6/tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
paddingSAME*
strides	
ј
=model_6/tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOpReadVariableOpFmodel_6_tubelet_embedding_33_conv3d_99_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ц
.model_6/tubelet_embedding_33/conv3d_99/BiasAddBiasAdd6model_6/tubelet_embedding_33/conv3d_99/Conv3D:output:0Emodel_6/tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 ™
+model_6/tubelet_embedding_33/conv3d_99/ReluRelu7model_6/tubelet_embedding_33/conv3d_99/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
 ы
7model_6/tubelet_embedding_33/max_pooling3d_66/MaxPool3D	MaxPool3D9model_6/tubelet_embedding_33/conv3d_99/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
ksize	
*
paddingVALID*
strides	
–
=model_6/tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOpReadVariableOpFmodel_6_tubelet_embedding_33_conv3d_100_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0®
.model_6/tubelet_embedding_33/conv3d_100/Conv3DConv3D@model_6/tubelet_embedding_33/max_pooling3d_66/MaxPool3D:output:0Emodel_6/tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
paddingSAME*
strides	
¬
>model_6/tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOpReadVariableOpGmodel_6_tubelet_embedding_33_conv3d_100_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0щ
/model_6/tubelet_embedding_33/conv3d_100/BiasAddBiasAdd7model_6/tubelet_embedding_33/conv3d_100/Conv3D:output:0Fmodel_6/tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@ђ
,model_6/tubelet_embedding_33/conv3d_100/ReluRelu8model_6/tubelet_embedding_33/conv3d_100/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
@ь
7model_6/tubelet_embedding_33/max_pooling3d_67/MaxPool3D	MaxPool3D:model_6/tubelet_embedding_33/conv3d_100/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
ksize	
*
paddingVALID*
strides	
—
=model_6/tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOpReadVariableOpFmodel_6_tubelet_embedding_33_conv3d_101_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0©
.model_6/tubelet_embedding_33/conv3d_101/Conv3DConv3D@model_6/tubelet_embedding_33/max_pooling3d_67/MaxPool3D:output:0Emodel_6/tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
paddingSAME*
strides	
√
>model_6/tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOpReadVariableOpGmodel_6_tubelet_embedding_33_conv3d_101_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ъ
/model_6/tubelet_embedding_33/conv3d_101/BiasAddBiasAdd7model_6/tubelet_embedding_33/conv3d_101/Conv3D:output:0Fmodel_6/tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А€
;model_6/tubelet_embedding_33/average_pooling3d_33/AvgPool3D	AvgPool3D8model_6/tubelet_embedding_33/conv3d_101/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
ksize	
*
paddingVALID*
strides	
°
-model_6/tubelet_embedding_33/reshape_33/ShapeShapeDmodel_6/tubelet_embedding_33/average_pooling3d_33/AvgPool3D:output:0*
T0*
_output_shapes
:Е
;model_6/tubelet_embedding_33/reshape_33/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: З
=model_6/tubelet_embedding_33/reshape_33/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:З
=model_6/tubelet_embedding_33/reshape_33/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Щ
5model_6/tubelet_embedding_33/reshape_33/strided_sliceStridedSlice6model_6/tubelet_embedding_33/reshape_33/Shape:output:0Dmodel_6/tubelet_embedding_33/reshape_33/strided_slice/stack:output:0Fmodel_6/tubelet_embedding_33/reshape_33/strided_slice/stack_1:output:0Fmodel_6/tubelet_embedding_33/reshape_33/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskВ
7model_6/tubelet_embedding_33/reshape_33/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€z
7model_6/tubelet_embedding_33/reshape_33/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Аѓ
5model_6/tubelet_embedding_33/reshape_33/Reshape/shapePack>model_6/tubelet_embedding_33/reshape_33/strided_slice:output:0@model_6/tubelet_embedding_33/reshape_33/Reshape/shape/1:output:0@model_6/tubelet_embedding_33/reshape_33/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:ч
/model_6/tubelet_embedding_33/reshape_33/ReshapeReshapeDmodel_6/tubelet_embedding_33/average_pooling3d_33/AvgPool3D:output:0>model_6/tubelet_embedding_33/reshape_33/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аd
"model_6/concatenate_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :К
model_6/concatenate_13/concatConcatV28model_6/tubelet_embedding_32/reshape_32/Reshape:output:08model_6/tubelet_embedding_33/reshape_33/Reshape:output:0+model_6/concatenate_13/concat/axis:output:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А√
8model_6/positional_encoder_16/embedding/embedding_lookupResourceGather?model_6_positional_encoder_16_embedding_embedding_lookup_494046$model_6_positional_encoder_16_494044*
Tindices0*R
_classH
FDloc:@model_6/positional_encoder_16/embedding/embedding_lookup/494046*
_output_shapes
:	(А*
dtype0О
Amodel_6/positional_encoder_16/embedding/embedding_lookup/IdentityIdentityAmodel_6/positional_encoder_16/embedding/embedding_lookup:output:0*
T0*R
_classH
FDloc:@model_6/positional_encoder_16/embedding/embedding_lookup/494046*
_output_shapes
:	(А≈
Cmodel_6/positional_encoder_16/embedding/embedding_lookup/Identity_1IdentityJmodel_6/positional_encoder_16/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	(А„
!model_6/positional_encoder_16/addAddV2&model_6/concatenate_13/concat:output:0Lmodel_6/positional_encoder_16/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АЗ
=model_6/layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:й
+model_6/layer_normalization_18/moments/meanMean%model_6/positional_encoder_16/add:z:0Fmodel_6/layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(ѓ
3model_6/layer_normalization_18/moments/StopGradientStopGradient4model_6/layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(й
8model_6/layer_normalization_18/moments/SquaredDifferenceSquaredDifference%model_6/positional_encoder_16/add:z:0<model_6/layer_normalization_18/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АЛ
Amodel_6/layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:И
/model_6/layer_normalization_18/moments/varianceMean<model_6/layer_normalization_18/moments/SquaredDifference:z:0Jmodel_6/layer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(s
.model_6/layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5ё
,model_6/layer_normalization_18/batchnorm/addAddV28model_6/layer_normalization_18/moments/variance:output:07model_6/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(Я
.model_6/layer_normalization_18/batchnorm/RsqrtRsqrt0model_6/layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(љ
;model_6/layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_6_layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0г
,model_6/layer_normalization_18/batchnorm/mulMul2model_6/layer_normalization_18/batchnorm/Rsqrt:y:0Cmodel_6/layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(А≈
.model_6/layer_normalization_18/batchnorm/mul_1Mul%model_6/positional_encoder_16/add:z:00model_6/layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А‘
.model_6/layer_normalization_18/batchnorm/mul_2Mul4model_6/layer_normalization_18/moments/mean:output:00model_6/layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аµ
7model_6/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp@model_6_layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0я
,model_6/layer_normalization_18/batchnorm/subSub?model_6/layer_normalization_18/batchnorm/ReadVariableOp:value:02model_6/layer_normalization_18/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А‘
.model_6/layer_normalization_18/batchnorm/add_1AddV22model_6/layer_normalization_18/batchnorm/mul_1:z:00model_6/layer_normalization_18/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А“
Amodel_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpJmodel_6_multi_head_attention_6_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0Ь
2model_6/multi_head_attention_6/query/einsum/EinsumEinsum2model_6/layer_normalization_18/batchnorm/add_1:z:0Imodel_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abdeє
7model_6/multi_head_attention_6/query/add/ReadVariableOpReadVariableOp@model_6_multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes
:	А*
dtype0к
(model_6/multi_head_attention_6/query/addAddV2;model_6/multi_head_attention_6/query/einsum/Einsum:output:0?model_6/multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(Аќ
?model_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_6_multi_head_attention_6_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0Ш
0model_6/multi_head_attention_6/key/einsum/EinsumEinsum2model_6/layer_normalization_18/batchnorm/add_1:z:0Gmodel_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abdeµ
5model_6/multi_head_attention_6/key/add/ReadVariableOpReadVariableOp>model_6_multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes
:	А*
dtype0д
&model_6/multi_head_attention_6/key/addAddV29model_6/multi_head_attention_6/key/einsum/Einsum:output:0=model_6/multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(А“
Amodel_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpJmodel_6_multi_head_attention_6_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0Ь
2model_6/multi_head_attention_6/value/einsum/EinsumEinsum2model_6/layer_normalization_18/batchnorm/add_1:z:0Imodel_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abdeє
7model_6/multi_head_attention_6/value/add/ReadVariableOpReadVariableOp@model_6_multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes
:	А*
dtype0к
(model_6/multi_head_attention_6/value/addAddV2;model_6/multi_head_attention_6/value/einsum/Einsum:output:0?model_6/multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(Аi
$model_6/multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=Ѕ
"model_6/multi_head_attention_6/MulMul,model_6/multi_head_attention_6/query/add:z:0-model_6/multi_head_attention_6/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€(Ам
,model_6/multi_head_attention_6/einsum/EinsumEinsum*model_6/multi_head_attention_6/key/add:z:0&model_6/multi_head_attention_6/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€((*
equationaecd,abcd->acbe™
.model_6/multi_head_attention_6/softmax/SoftmaxSoftmax5model_6/multi_head_attention_6/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€((ѓ
/model_6/multi_head_attention_6/dropout/IdentityIdentity8model_6/multi_head_attention_6/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€((Г
.model_6/multi_head_attention_6/einsum_1/EinsumEinsum8model_6/multi_head_attention_6/dropout/Identity:output:0,model_6/multi_head_attention_6/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationacbe,aecd->abcdи
Lmodel_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpUmodel_6_multi_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0≥
=model_6/multi_head_attention_6/attention_output/einsum/EinsumEinsum7model_6/multi_head_attention_6/einsum_1/Einsum:output:0Tmodel_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А*
equationabcd,cde->abeЋ
Bmodel_6/multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpKmodel_6_multi_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype0З
3model_6/multi_head_attention_6/attention_output/addAddV2Fmodel_6/multi_head_attention_6/attention_output/einsum/Einsum:output:0Jmodel_6/multi_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(А≤
model_6/add_12/addAddV27model_6/multi_head_attention_6/attention_output/add:z:0%model_6/positional_encoder_16/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЗ
=model_6/layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Џ
+model_6/layer_normalization_19/moments/meanMeanmodel_6/add_12/add:z:0Fmodel_6/layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(ѓ
3model_6/layer_normalization_19/moments/StopGradientStopGradient4model_6/layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(Џ
8model_6/layer_normalization_19/moments/SquaredDifferenceSquaredDifferencemodel_6/add_12/add:z:0<model_6/layer_normalization_19/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АЛ
Amodel_6/layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:И
/model_6/layer_normalization_19/moments/varianceMean<model_6/layer_normalization_19/moments/SquaredDifference:z:0Jmodel_6/layer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(s
.model_6/layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5ё
,model_6/layer_normalization_19/batchnorm/addAddV28model_6/layer_normalization_19/moments/variance:output:07model_6/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(Я
.model_6/layer_normalization_19/batchnorm/RsqrtRsqrt0model_6/layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(љ
;model_6/layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_6_layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0г
,model_6/layer_normalization_19/batchnorm/mulMul2model_6/layer_normalization_19/batchnorm/Rsqrt:y:0Cmodel_6/layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аґ
.model_6/layer_normalization_19/batchnorm/mul_1Mulmodel_6/add_12/add:z:00model_6/layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А‘
.model_6/layer_normalization_19/batchnorm/mul_2Mul4model_6/layer_normalization_19/moments/mean:output:00model_6/layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аµ
7model_6/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp@model_6_layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0я
,model_6/layer_normalization_19/batchnorm/subSub?model_6/layer_normalization_19/batchnorm/ReadVariableOp:value:02model_6/layer_normalization_19/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А‘
.model_6/layer_normalization_19/batchnorm/add_1AddV22model_6/layer_normalization_19/batchnorm/mul_1:z:00model_6/layer_normalization_19/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЄ
6model_6/sequential_6/dense_12/Tensordot/ReadVariableOpReadVariableOp?model_6_sequential_6_dense_12_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype0v
,model_6/sequential_6/dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,model_6/sequential_6/dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       П
-model_6/sequential_6/dense_12/Tensordot/ShapeShape2model_6/layer_normalization_19/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
5model_6/sequential_6/dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ≥
0model_6/sequential_6/dense_12/Tensordot/GatherV2GatherV26model_6/sequential_6/dense_12/Tensordot/Shape:output:05model_6/sequential_6/dense_12/Tensordot/free:output:0>model_6/sequential_6/dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7model_6/sequential_6/dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ј
2model_6/sequential_6/dense_12/Tensordot/GatherV2_1GatherV26model_6/sequential_6/dense_12/Tensordot/Shape:output:05model_6/sequential_6/dense_12/Tensordot/axes:output:0@model_6/sequential_6/dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-model_6/sequential_6/dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: »
,model_6/sequential_6/dense_12/Tensordot/ProdProd9model_6/sequential_6/dense_12/Tensordot/GatherV2:output:06model_6/sequential_6/dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/model_6/sequential_6/dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ќ
.model_6/sequential_6/dense_12/Tensordot/Prod_1Prod;model_6/sequential_6/dense_12/Tensordot/GatherV2_1:output:08model_6/sequential_6/dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3model_6/sequential_6/dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.model_6/sequential_6/dense_12/Tensordot/concatConcatV25model_6/sequential_6/dense_12/Tensordot/free:output:05model_6/sequential_6/dense_12/Tensordot/axes:output:0<model_6/sequential_6/dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:”
-model_6/sequential_6/dense_12/Tensordot/stackPack5model_6/sequential_6/dense_12/Tensordot/Prod:output:07model_6/sequential_6/dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:в
1model_6/sequential_6/dense_12/Tensordot/transpose	Transpose2model_6/layer_normalization_19/batchnorm/add_1:z:07model_6/sequential_6/dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Ад
/model_6/sequential_6/dense_12/Tensordot/ReshapeReshape5model_6/sequential_6/dense_12/Tensordot/transpose:y:06model_6/sequential_6/dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€е
.model_6/sequential_6/dense_12/Tensordot/MatMulMatMul8model_6/sequential_6/dense_12/Tensordot/Reshape:output:0>model_6/sequential_6/dense_12/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аz
/model_6/sequential_6/dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аw
5model_6/sequential_6/dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0model_6/sequential_6/dense_12/Tensordot/concat_1ConcatV29model_6/sequential_6/dense_12/Tensordot/GatherV2:output:08model_6/sequential_6/dense_12/Tensordot/Const_2:output:0>model_6/sequential_6/dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ё
'model_6/sequential_6/dense_12/TensordotReshape8model_6/sequential_6/dense_12/Tensordot/MatMul:product:09model_6/sequential_6/dense_12/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аѓ
4model_6/sequential_6/dense_12/BiasAdd/ReadVariableOpReadVariableOp=model_6_sequential_6_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0„
%model_6/sequential_6/dense_12/BiasAddBiasAdd0model_6/sequential_6/dense_12/Tensordot:output:0<model_6/sequential_6/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аm
(model_6/sequential_6/dense_12/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?«
&model_6/sequential_6/dense_12/Gelu/mulMul1model_6/sequential_6/dense_12/Gelu/mul/x:output:0.model_6/sequential_6/dense_12/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аn
)model_6/sequential_6/dense_12/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?–
*model_6/sequential_6/dense_12/Gelu/truedivRealDiv.model_6/sequential_6/dense_12/BiasAdd:output:02model_6/sequential_6/dense_12/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АФ
&model_6/sequential_6/dense_12/Gelu/ErfErf.model_6/sequential_6/dense_12/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аm
(model_6/sequential_6/dense_12/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≈
&model_6/sequential_6/dense_12/Gelu/addAddV21model_6/sequential_6/dense_12/Gelu/add/x:output:0*model_6/sequential_6/dense_12/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€(АЊ
(model_6/sequential_6/dense_12/Gelu/mul_1Mul*model_6/sequential_6/dense_12/Gelu/mul:z:0*model_6/sequential_6/dense_12/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АШ
model_6/add_13/addAddV2,model_6/sequential_6/dense_12/Gelu/mul_1:z:0model_6/add_12/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЗ
=model_6/layer_normalization_20/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Џ
+model_6/layer_normalization_20/moments/meanMeanmodel_6/add_13/add:z:0Fmodel_6/layer_normalization_20/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(ѓ
3model_6/layer_normalization_20/moments/StopGradientStopGradient4model_6/layer_normalization_20/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(Џ
8model_6/layer_normalization_20/moments/SquaredDifferenceSquaredDifferencemodel_6/add_13/add:z:0<model_6/layer_normalization_20/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АЛ
Amodel_6/layer_normalization_20/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:И
/model_6/layer_normalization_20/moments/varianceMean<model_6/layer_normalization_20/moments/SquaredDifference:z:0Jmodel_6/layer_normalization_20/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(s
.model_6/layer_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5ё
,model_6/layer_normalization_20/batchnorm/addAddV28model_6/layer_normalization_20/moments/variance:output:07model_6/layer_normalization_20/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(Я
.model_6/layer_normalization_20/batchnorm/RsqrtRsqrt0model_6/layer_normalization_20/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(љ
;model_6/layer_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_6_layer_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0г
,model_6/layer_normalization_20/batchnorm/mulMul2model_6/layer_normalization_20/batchnorm/Rsqrt:y:0Cmodel_6/layer_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аґ
.model_6/layer_normalization_20/batchnorm/mul_1Mulmodel_6/add_13/add:z:00model_6/layer_normalization_20/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А‘
.model_6/layer_normalization_20/batchnorm/mul_2Mul4model_6/layer_normalization_20/moments/mean:output:00model_6/layer_normalization_20/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аµ
7model_6/layer_normalization_20/batchnorm/ReadVariableOpReadVariableOp@model_6_layer_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0я
,model_6/layer_normalization_20/batchnorm/subSub?model_6/layer_normalization_20/batchnorm/ReadVariableOp:value:02model_6/layer_normalization_20/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А‘
.model_6/layer_normalization_20/batchnorm/add_1AddV22model_6/layer_normalization_20/batchnorm/mul_1:z:00model_6/layer_normalization_20/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А{
9model_6/global_average_pooling1d_6/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Џ
'model_6/global_average_pooling1d_6/MeanMean2model_6/layer_normalization_20/batchnorm/add_1:z:0Bmodel_6/global_average_pooling1d_6/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЧ
&model_6/dense_13/MatMul/ReadVariableOpReadVariableOp/model_6_dense_13_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0µ
model_6/dense_13/MatMulMatMul0model_6/global_average_pooling1d_6/Mean:output:0.model_6/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
'model_6/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model_6/dense_13/BiasAddBiasAdd!model_6/dense_13/MatMul:product:0/model_6/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€p
IdentityIdentity!model_6/dense_13/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€і
NoOpNoOp(^model_6/dense_13/BiasAdd/ReadVariableOp'^model_6/dense_13/MatMul/ReadVariableOp8^model_6/layer_normalization_18/batchnorm/ReadVariableOp<^model_6/layer_normalization_18/batchnorm/mul/ReadVariableOp8^model_6/layer_normalization_19/batchnorm/ReadVariableOp<^model_6/layer_normalization_19/batchnorm/mul/ReadVariableOp8^model_6/layer_normalization_20/batchnorm/ReadVariableOp<^model_6/layer_normalization_20/batchnorm/mul/ReadVariableOpC^model_6/multi_head_attention_6/attention_output/add/ReadVariableOpM^model_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp6^model_6/multi_head_attention_6/key/add/ReadVariableOp@^model_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp8^model_6/multi_head_attention_6/query/add/ReadVariableOpB^model_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp8^model_6/multi_head_attention_6/value/add/ReadVariableOpB^model_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp9^model_6/positional_encoder_16/embedding/embedding_lookup5^model_6/sequential_6/dense_12/BiasAdd/ReadVariableOp7^model_6/sequential_6/dense_12/Tensordot/ReadVariableOp>^model_6/tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp=^model_6/tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp>^model_6/tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp=^model_6/tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp>^model_6/tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp=^model_6/tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp?^model_6/tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp>^model_6/tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp?^model_6/tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp>^model_6/tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp>^model_6/tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp=^model_6/tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2R
'model_6/dense_13/BiasAdd/ReadVariableOp'model_6/dense_13/BiasAdd/ReadVariableOp2P
&model_6/dense_13/MatMul/ReadVariableOp&model_6/dense_13/MatMul/ReadVariableOp2r
7model_6/layer_normalization_18/batchnorm/ReadVariableOp7model_6/layer_normalization_18/batchnorm/ReadVariableOp2z
;model_6/layer_normalization_18/batchnorm/mul/ReadVariableOp;model_6/layer_normalization_18/batchnorm/mul/ReadVariableOp2r
7model_6/layer_normalization_19/batchnorm/ReadVariableOp7model_6/layer_normalization_19/batchnorm/ReadVariableOp2z
;model_6/layer_normalization_19/batchnorm/mul/ReadVariableOp;model_6/layer_normalization_19/batchnorm/mul/ReadVariableOp2r
7model_6/layer_normalization_20/batchnorm/ReadVariableOp7model_6/layer_normalization_20/batchnorm/ReadVariableOp2z
;model_6/layer_normalization_20/batchnorm/mul/ReadVariableOp;model_6/layer_normalization_20/batchnorm/mul/ReadVariableOp2И
Bmodel_6/multi_head_attention_6/attention_output/add/ReadVariableOpBmodel_6/multi_head_attention_6/attention_output/add/ReadVariableOp2Ь
Lmodel_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpLmodel_6/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2n
5model_6/multi_head_attention_6/key/add/ReadVariableOp5model_6/multi_head_attention_6/key/add/ReadVariableOp2В
?model_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp?model_6/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2r
7model_6/multi_head_attention_6/query/add/ReadVariableOp7model_6/multi_head_attention_6/query/add/ReadVariableOp2Ж
Amodel_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpAmodel_6/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2r
7model_6/multi_head_attention_6/value/add/ReadVariableOp7model_6/multi_head_attention_6/value/add/ReadVariableOp2Ж
Amodel_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpAmodel_6/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2t
8model_6/positional_encoder_16/embedding/embedding_lookup8model_6/positional_encoder_16/embedding/embedding_lookup2l
4model_6/sequential_6/dense_12/BiasAdd/ReadVariableOp4model_6/sequential_6/dense_12/BiasAdd/ReadVariableOp2p
6model_6/sequential_6/dense_12/Tensordot/ReadVariableOp6model_6/sequential_6/dense_12/Tensordot/ReadVariableOp2~
=model_6/tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp=model_6/tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp2|
<model_6/tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp<model_6/tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp2~
=model_6/tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp=model_6/tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp2|
<model_6/tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp<model_6/tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp2~
=model_6/tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp=model_6/tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp2|
<model_6/tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp<model_6/tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp2А
>model_6/tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp>model_6/tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp2~
=model_6/tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp=model_6/tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp2А
>model_6/tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp>model_6/tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp2~
=model_6/tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp=model_6/tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp2~
=model_6/tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp=model_6/tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp2|
<model_6/tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp<model_6/tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp:] Y
3
_output_shapes!
:€€€€€€€€€

"
_user_specified_name
input_17: 

_output_shapes
:(
ю
У
R__inference_layer_normalization_20_layer_call_and_return_conditional_losses_494685

inputs4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(М
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ж
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0В
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АА
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
—
[
/__inference_concatenate_13_layer_call_fn_496214
inputs_0
inputs_1
identityћ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *S
fNRL
J__inference_concatenate_13_layer_call_and_return_conditional_losses_494511e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€(А:€€€€€€€€€(А:V R
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/1
Ђ„
Ћ!
C__inference_model_6_layer_call_and_return_conditional_losses_496031

inputs[
=tubelet_embedding_32_conv3d_96_conv3d_readvariableop_resource: L
>tubelet_embedding_32_conv3d_96_biasadd_readvariableop_resource: [
=tubelet_embedding_32_conv3d_97_conv3d_readvariableop_resource: @L
>tubelet_embedding_32_conv3d_97_biasadd_readvariableop_resource:@\
=tubelet_embedding_32_conv3d_98_conv3d_readvariableop_resource:@АM
>tubelet_embedding_32_conv3d_98_biasadd_readvariableop_resource:	А[
=tubelet_embedding_33_conv3d_99_conv3d_readvariableop_resource: L
>tubelet_embedding_33_conv3d_99_biasadd_readvariableop_resource: \
>tubelet_embedding_33_conv3d_100_conv3d_readvariableop_resource: @M
?tubelet_embedding_33_conv3d_100_biasadd_readvariableop_resource:@]
>tubelet_embedding_33_conv3d_101_conv3d_readvariableop_resource:@АN
?tubelet_embedding_33_conv3d_101_biasadd_readvariableop_resource:	А 
positional_encoder_16_495887J
7positional_encoder_16_embedding_embedding_lookup_495889:	(АK
<layer_normalization_18_batchnorm_mul_readvariableop_resource:	АG
8layer_normalization_18_batchnorm_readvariableop_resource:	АZ
Bmulti_head_attention_6_query_einsum_einsum_readvariableop_resource:ААK
8multi_head_attention_6_query_add_readvariableop_resource:	АX
@multi_head_attention_6_key_einsum_einsum_readvariableop_resource:ААI
6multi_head_attention_6_key_add_readvariableop_resource:	АZ
Bmulti_head_attention_6_value_einsum_einsum_readvariableop_resource:ААK
8multi_head_attention_6_value_add_readvariableop_resource:	Аe
Mmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource:ААR
Cmulti_head_attention_6_attention_output_add_readvariableop_resource:	АK
<layer_normalization_19_batchnorm_mul_readvariableop_resource:	АG
8layer_normalization_19_batchnorm_readvariableop_resource:	АK
7sequential_6_dense_12_tensordot_readvariableop_resource:
ААD
5sequential_6_dense_12_biasadd_readvariableop_resource:	АK
<layer_normalization_20_batchnorm_mul_readvariableop_resource:	АG
8layer_normalization_20_batchnorm_readvariableop_resource:	А:
'dense_13_matmul_readvariableop_resource:	А6
(dense_13_biasadd_readvariableop_resource:
identityИҐdense_13/BiasAdd/ReadVariableOpҐdense_13/MatMul/ReadVariableOpҐ/layer_normalization_18/batchnorm/ReadVariableOpҐ3layer_normalization_18/batchnorm/mul/ReadVariableOpҐ/layer_normalization_19/batchnorm/ReadVariableOpҐ3layer_normalization_19/batchnorm/mul/ReadVariableOpҐ/layer_normalization_20/batchnorm/ReadVariableOpҐ3layer_normalization_20/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_6/attention_output/add/ReadVariableOpҐDmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_6/key/add/ReadVariableOpҐ7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_6/query/add/ReadVariableOpҐ9multi_head_attention_6/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_6/value/add/ReadVariableOpҐ9multi_head_attention_6/value/einsum/Einsum/ReadVariableOpҐ0positional_encoder_16/embedding/embedding_lookupҐ,sequential_6/dense_12/BiasAdd/ReadVariableOpҐ.sequential_6/dense_12/Tensordot/ReadVariableOpҐ5tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOpҐ4tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOpҐ5tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOpҐ4tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOpҐ5tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOpҐ4tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOpҐ6tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOpҐ5tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOpҐ6tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOpҐ5tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOpҐ5tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOpҐ4tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOpМ
/tf.__operators__.getitem_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               а
)tf.__operators__.getitem_37/strided_sliceStridedSliceinputs8tf.__operators__.getitem_37/strided_slice/stack:output:0:tf.__operators__.getitem_37/strided_slice/stack_1:output:0:tf.__operators__.getitem_37/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_maskМ
/tf.__operators__.getitem_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               а
)tf.__operators__.getitem_36/strided_sliceStridedSliceinputs8tf.__operators__.getitem_36/strided_slice/stack:output:0:tf.__operators__.getitem_36/strided_slice/stack_1:output:0:tf.__operators__.getitem_36/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_maskЊ
4tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_32_conv3d_96_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0И
%tubelet_embedding_32/conv3d_96/Conv3DConv3D2tf.__operators__.getitem_36/strided_slice:output:0<tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
paddingSAME*
strides	
∞
5tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_32_conv3d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ё
&tubelet_embedding_32/conv3d_96/BiasAddBiasAdd.tubelet_embedding_32/conv3d_96/Conv3D:output:0=tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 Ъ
#tubelet_embedding_32/conv3d_96/ReluRelu/tubelet_embedding_32/conv3d_96/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
 л
/tubelet_embedding_32/max_pooling3d_64/MaxPool3D	MaxPool3D1tubelet_embedding_32/conv3d_96/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
ksize	
*
paddingVALID*
strides	
Њ
4tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_32_conv3d_97_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0О
%tubelet_embedding_32/conv3d_97/Conv3DConv3D8tubelet_embedding_32/max_pooling3d_64/MaxPool3D:output:0<tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
paddingSAME*
strides	
∞
5tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_32_conv3d_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ё
&tubelet_embedding_32/conv3d_97/BiasAddBiasAdd.tubelet_embedding_32/conv3d_97/Conv3D:output:0=tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@Ъ
#tubelet_embedding_32/conv3d_97/ReluRelu/tubelet_embedding_32/conv3d_97/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
@л
/tubelet_embedding_32/max_pooling3d_65/MaxPool3D	MaxPool3D1tubelet_embedding_32/conv3d_97/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
ksize	
*
paddingVALID*
strides	
њ
4tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_32_conv3d_98_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0П
%tubelet_embedding_32/conv3d_98/Conv3DConv3D8tubelet_embedding_32/max_pooling3d_65/MaxPool3D:output:0<tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
paddingSAME*
strides	
±
5tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_32_conv3d_98_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0я
&tubelet_embedding_32/conv3d_98/BiasAddBiasAdd.tubelet_embedding_32/conv3d_98/Conv3D:output:0=tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
Ао
3tubelet_embedding_32/average_pooling3d_32/AvgPool3D	AvgPool3D/tubelet_embedding_32/conv3d_98/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
ksize	
*
paddingVALID*
strides	
С
%tubelet_embedding_32/reshape_32/ShapeShape<tubelet_embedding_32/average_pooling3d_32/AvgPool3D:output:0*
T0*
_output_shapes
:}
3tubelet_embedding_32/reshape_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tubelet_embedding_32/reshape_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tubelet_embedding_32/reshape_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-tubelet_embedding_32/reshape_32/strided_sliceStridedSlice.tubelet_embedding_32/reshape_32/Shape:output:0<tubelet_embedding_32/reshape_32/strided_slice/stack:output:0>tubelet_embedding_32/reshape_32/strided_slice/stack_1:output:0>tubelet_embedding_32/reshape_32/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/tubelet_embedding_32/reshape_32/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€r
/tubelet_embedding_32/reshape_32/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АП
-tubelet_embedding_32/reshape_32/Reshape/shapePack6tubelet_embedding_32/reshape_32/strided_slice:output:08tubelet_embedding_32/reshape_32/Reshape/shape/1:output:08tubelet_embedding_32/reshape_32/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:я
'tubelet_embedding_32/reshape_32/ReshapeReshape<tubelet_embedding_32/average_pooling3d_32/AvgPool3D:output:06tubelet_embedding_32/reshape_32/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АЊ
4tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_33_conv3d_99_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0И
%tubelet_embedding_33/conv3d_99/Conv3DConv3D2tf.__operators__.getitem_37/strided_slice:output:0<tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
paddingSAME*
strides	
∞
5tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_33_conv3d_99_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ё
&tubelet_embedding_33/conv3d_99/BiasAddBiasAdd.tubelet_embedding_33/conv3d_99/Conv3D:output:0=tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 Ъ
#tubelet_embedding_33/conv3d_99/ReluRelu/tubelet_embedding_33/conv3d_99/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
 л
/tubelet_embedding_33/max_pooling3d_66/MaxPool3D	MaxPool3D1tubelet_embedding_33/conv3d_99/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
ksize	
*
paddingVALID*
strides	
ј
5tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOpReadVariableOp>tubelet_embedding_33_conv3d_100_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0Р
&tubelet_embedding_33/conv3d_100/Conv3DConv3D8tubelet_embedding_33/max_pooling3d_66/MaxPool3D:output:0=tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
paddingSAME*
strides	
≤
6tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOpReadVariableOp?tubelet_embedding_33_conv3d_100_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0б
'tubelet_embedding_33/conv3d_100/BiasAddBiasAdd/tubelet_embedding_33/conv3d_100/Conv3D:output:0>tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@Ь
$tubelet_embedding_33/conv3d_100/ReluRelu0tubelet_embedding_33/conv3d_100/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
@м
/tubelet_embedding_33/max_pooling3d_67/MaxPool3D	MaxPool3D2tubelet_embedding_33/conv3d_100/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
ksize	
*
paddingVALID*
strides	
Ѕ
5tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOpReadVariableOp>tubelet_embedding_33_conv3d_101_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0С
&tubelet_embedding_33/conv3d_101/Conv3DConv3D8tubelet_embedding_33/max_pooling3d_67/MaxPool3D:output:0=tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
paddingSAME*
strides	
≥
6tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOpReadVariableOp?tubelet_embedding_33_conv3d_101_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0в
'tubelet_embedding_33/conv3d_101/BiasAddBiasAdd/tubelet_embedding_33/conv3d_101/Conv3D:output:0>tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
Ап
3tubelet_embedding_33/average_pooling3d_33/AvgPool3D	AvgPool3D0tubelet_embedding_33/conv3d_101/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
ksize	
*
paddingVALID*
strides	
С
%tubelet_embedding_33/reshape_33/ShapeShape<tubelet_embedding_33/average_pooling3d_33/AvgPool3D:output:0*
T0*
_output_shapes
:}
3tubelet_embedding_33/reshape_33/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tubelet_embedding_33/reshape_33/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tubelet_embedding_33/reshape_33/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-tubelet_embedding_33/reshape_33/strided_sliceStridedSlice.tubelet_embedding_33/reshape_33/Shape:output:0<tubelet_embedding_33/reshape_33/strided_slice/stack:output:0>tubelet_embedding_33/reshape_33/strided_slice/stack_1:output:0>tubelet_embedding_33/reshape_33/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/tubelet_embedding_33/reshape_33/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€r
/tubelet_embedding_33/reshape_33/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АП
-tubelet_embedding_33/reshape_33/Reshape/shapePack6tubelet_embedding_33/reshape_33/strided_slice:output:08tubelet_embedding_33/reshape_33/Reshape/shape/1:output:08tubelet_embedding_33/reshape_33/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:я
'tubelet_embedding_33/reshape_33/ReshapeReshape<tubelet_embedding_33/average_pooling3d_33/AvgPool3D:output:06tubelet_embedding_33/reshape_33/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А\
concatenate_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :к
concatenate_13/concatConcatV20tubelet_embedding_32/reshape_32/Reshape:output:00tubelet_embedding_33/reshape_33/Reshape:output:0#concatenate_13/concat/axis:output:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А£
0positional_encoder_16/embedding/embedding_lookupResourceGather7positional_encoder_16_embedding_embedding_lookup_495889positional_encoder_16_495887*
Tindices0*J
_class@
><loc:@positional_encoder_16/embedding/embedding_lookup/495889*
_output_shapes
:	(А*
dtype0ц
9positional_encoder_16/embedding/embedding_lookup/IdentityIdentity9positional_encoder_16/embedding/embedding_lookup:output:0*
T0*J
_class@
><loc:@positional_encoder_16/embedding/embedding_lookup/495889*
_output_shapes
:	(Аµ
;positional_encoder_16/embedding/embedding_lookup/Identity_1IdentityBpositional_encoder_16/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	(Ањ
positional_encoder_16/addAddV2concatenate_13/concat:output:0Dpositional_encoder_16/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А
5layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:—
#layer_normalization_18/moments/meanMeanpositional_encoder_16/add:z:0>layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(Я
+layer_normalization_18/moments/StopGradientStopGradient,layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(—
0layer_normalization_18/moments/SquaredDifferenceSquaredDifferencepositional_encoder_16/add:z:04layer_normalization_18/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АГ
9layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:р
'layer_normalization_18/moments/varianceMean4layer_normalization_18/moments/SquaredDifference:z:0Blayer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(k
&layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5∆
$layer_normalization_18/batchnorm/addAddV20layer_normalization_18/moments/variance:output:0/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(П
&layer_normalization_18/batchnorm/RsqrtRsqrt(layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(≠
3layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ћ
$layer_normalization_18/batchnorm/mulMul*layer_normalization_18/batchnorm/Rsqrt:y:0;layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(А≠
&layer_normalization_18/batchnorm/mul_1Mulpositional_encoder_16/add:z:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_18/batchnorm/mul_2Mul,layer_normalization_18/moments/mean:output:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А•
/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0«
$layer_normalization_18/batchnorm/subSub7layer_normalization_18/batchnorm/ReadVariableOp:value:0*layer_normalization_18/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_18/batchnorm/add_1AddV2*layer_normalization_18/batchnorm/mul_1:z:0(layer_normalization_18/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А¬
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0Д
*multi_head_attention_6/query/einsum/EinsumEinsum*layer_normalization_18/batchnorm/add_1:z:0Amulti_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde©
/multi_head_attention_6/query/add/ReadVariableOpReadVariableOp8multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes
:	А*
dtype0“
 multi_head_attention_6/query/addAddV23multi_head_attention_6/query/einsum/Einsum:output:07multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АЊ
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_6_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0А
(multi_head_attention_6/key/einsum/EinsumEinsum*layer_normalization_18/batchnorm/add_1:z:0?multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde•
-multi_head_attention_6/key/add/ReadVariableOpReadVariableOp6multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes
:	А*
dtype0ћ
multi_head_attention_6/key/addAddV21multi_head_attention_6/key/einsum/Einsum:output:05multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(А¬
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0Д
*multi_head_attention_6/value/einsum/EinsumEinsum*layer_normalization_18/batchnorm/add_1:z:0Amulti_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde©
/multi_head_attention_6/value/add/ReadVariableOpReadVariableOp8multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes
:	А*
dtype0“
 multi_head_attention_6/value/addAddV23multi_head_attention_6/value/einsum/Einsum:output:07multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(Аa
multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=©
multi_head_attention_6/MulMul$multi_head_attention_6/query/add:z:0%multi_head_attention_6/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€(А‘
$multi_head_attention_6/einsum/EinsumEinsum"multi_head_attention_6/key/add:z:0multi_head_attention_6/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€((*
equationaecd,abcd->acbeЪ
&multi_head_attention_6/softmax/SoftmaxSoftmax-multi_head_attention_6/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€((q
,multi_head_attention_6/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?‘
*multi_head_attention_6/dropout/dropout/MulMul0multi_head_attention_6/softmax/Softmax:softmax:05multi_head_attention_6/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€((М
,multi_head_attention_6/dropout/dropout/ShapeShape0multi_head_attention_6/softmax/Softmax:softmax:0*
T0*
_output_shapes
:ё
Cmulti_head_attention_6/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_6/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€((*
dtype0*

seed*z
5multi_head_attention_6/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=Л
3multi_head_attention_6/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_6/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_6/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€((µ
+multi_head_attention_6/dropout/dropout/CastCast7multi_head_attention_6/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€((ќ
,multi_head_attention_6/dropout/dropout/Mul_1Mul.multi_head_attention_6/dropout/dropout/Mul:z:0/multi_head_attention_6/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€((л
&multi_head_attention_6/einsum_1/EinsumEinsum0multi_head_attention_6/dropout/dropout/Mul_1:z:0$multi_head_attention_6/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationacbe,aecd->abcdЎ
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0Ы
5multi_head_attention_6/attention_output/einsum/EinsumEinsum/multi_head_attention_6/einsum_1/Einsum:output:0Lmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А*
equationabcd,cde->abeї
:multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype0п
+multi_head_attention_6/attention_output/addAddV2>multi_head_attention_6/attention_output/einsum/Einsum:output:0Bmulti_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(АЪ

add_12/addAddV2/multi_head_attention_6/attention_output/add:z:0positional_encoder_16/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А
5layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
#layer_normalization_19/moments/meanMeanadd_12/add:z:0>layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(Я
+layer_normalization_19/moments/StopGradientStopGradient,layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(¬
0layer_normalization_19/moments/SquaredDifferenceSquaredDifferenceadd_12/add:z:04layer_normalization_19/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АГ
9layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:р
'layer_normalization_19/moments/varianceMean4layer_normalization_19/moments/SquaredDifference:z:0Blayer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(k
&layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5∆
$layer_normalization_19/batchnorm/addAddV20layer_normalization_19/moments/variance:output:0/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(П
&layer_normalization_19/batchnorm/RsqrtRsqrt(layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(≠
3layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ћ
$layer_normalization_19/batchnorm/mulMul*layer_normalization_19/batchnorm/Rsqrt:y:0;layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(АЮ
&layer_normalization_19/batchnorm/mul_1Muladd_12/add:z:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_19/batchnorm/mul_2Mul,layer_normalization_19/moments/mean:output:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А•
/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0«
$layer_normalization_19/batchnorm/subSub7layer_normalization_19/batchnorm/ReadVariableOp:value:0*layer_normalization_19/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_19/batchnorm/add_1AddV2*layer_normalization_19/batchnorm/mul_1:z:0(layer_normalization_19/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А®
.sequential_6/dense_12/Tensordot/ReadVariableOpReadVariableOp7sequential_6_dense_12_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype0n
$sequential_6/dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_6/dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_6/dense_12/Tensordot/ShapeShape*layer_normalization_19/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_6/dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : У
(sequential_6/dense_12/Tensordot/GatherV2GatherV2.sequential_6/dense_12/Tensordot/Shape:output:0-sequential_6/dense_12/Tensordot/free:output:06sequential_6/dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_6/dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
*sequential_6/dense_12/Tensordot/GatherV2_1GatherV2.sequential_6/dense_12/Tensordot/Shape:output:0-sequential_6/dense_12/Tensordot/axes:output:08sequential_6/dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_6/dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ∞
$sequential_6/dense_12/Tensordot/ProdProd1sequential_6/dense_12/Tensordot/GatherV2:output:0.sequential_6/dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_6/dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ґ
&sequential_6/dense_12/Tensordot/Prod_1Prod3sequential_6/dense_12/Tensordot/GatherV2_1:output:00sequential_6/dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_6/dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
&sequential_6/dense_12/Tensordot/concatConcatV2-sequential_6/dense_12/Tensordot/free:output:0-sequential_6/dense_12/Tensordot/axes:output:04sequential_6/dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ї
%sequential_6/dense_12/Tensordot/stackPack-sequential_6/dense_12/Tensordot/Prod:output:0/sequential_6/dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
)sequential_6/dense_12/Tensordot/transpose	Transpose*layer_normalization_19/batchnorm/add_1:z:0/sequential_6/dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аћ
'sequential_6/dense_12/Tensordot/ReshapeReshape-sequential_6/dense_12/Tensordot/transpose:y:0.sequential_6/dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Ќ
&sequential_6/dense_12/Tensordot/MatMulMatMul0sequential_6/dense_12/Tensordot/Reshape:output:06sequential_6/dense_12/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аr
'sequential_6/dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аo
-sequential_6/dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : €
(sequential_6/dense_12/Tensordot/concat_1ConcatV21sequential_6/dense_12/Tensordot/GatherV2:output:00sequential_6/dense_12/Tensordot/Const_2:output:06sequential_6/dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:∆
sequential_6/dense_12/TensordotReshape0sequential_6/dense_12/Tensordot/MatMul:product:01sequential_6/dense_12/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АЯ
,sequential_6/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0њ
sequential_6/dense_12/BiasAddBiasAdd(sequential_6/dense_12/Tensordot:output:04sequential_6/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аe
 sequential_6/dense_12/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ѓ
sequential_6/dense_12/Gelu/mulMul)sequential_6/dense_12/Gelu/mul/x:output:0&sequential_6/dense_12/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аf
!sequential_6/dense_12/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?Є
"sequential_6/dense_12/Gelu/truedivRealDiv&sequential_6/dense_12/BiasAdd:output:0*sequential_6/dense_12/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АД
sequential_6/dense_12/Gelu/ErfErf&sequential_6/dense_12/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аe
 sequential_6/dense_12/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≠
sequential_6/dense_12/Gelu/addAddV2)sequential_6/dense_12/Gelu/add/x:output:0"sequential_6/dense_12/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€(А¶
 sequential_6/dense_12/Gelu/mul_1Mul"sequential_6/dense_12/Gelu/mul:z:0"sequential_6/dense_12/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АА

add_13/addAddV2$sequential_6/dense_12/Gelu/mul_1:z:0add_12/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А
5layer_normalization_20/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
#layer_normalization_20/moments/meanMeanadd_13/add:z:0>layer_normalization_20/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(Я
+layer_normalization_20/moments/StopGradientStopGradient,layer_normalization_20/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(¬
0layer_normalization_20/moments/SquaredDifferenceSquaredDifferenceadd_13/add:z:04layer_normalization_20/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АГ
9layer_normalization_20/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:р
'layer_normalization_20/moments/varianceMean4layer_normalization_20/moments/SquaredDifference:z:0Blayer_normalization_20/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(k
&layer_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5∆
$layer_normalization_20/batchnorm/addAddV20layer_normalization_20/moments/variance:output:0/layer_normalization_20/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(П
&layer_normalization_20/batchnorm/RsqrtRsqrt(layer_normalization_20/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(≠
3layer_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ћ
$layer_normalization_20/batchnorm/mulMul*layer_normalization_20/batchnorm/Rsqrt:y:0;layer_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(АЮ
&layer_normalization_20/batchnorm/mul_1Muladd_13/add:z:0(layer_normalization_20/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_20/batchnorm/mul_2Mul,layer_normalization_20/moments/mean:output:0(layer_normalization_20/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А•
/layer_normalization_20/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0«
$layer_normalization_20/batchnorm/subSub7layer_normalization_20/batchnorm/ReadVariableOp:value:0*layer_normalization_20/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_20/batchnorm/add_1AddV2*layer_normalization_20/batchnorm/mul_1:z:0(layer_normalization_20/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аs
1global_average_pooling1d_6/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¬
global_average_pooling1d_6/MeanMean*layer_normalization_20/batchnorm/add_1:z:0:global_average_pooling1d_6/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЗ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Э
dense_13/MatMulMatMul(global_average_pooling1d_6/Mean:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_13/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Љ
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp0^layer_normalization_18/batchnorm/ReadVariableOp4^layer_normalization_18/batchnorm/mul/ReadVariableOp0^layer_normalization_19/batchnorm/ReadVariableOp4^layer_normalization_19/batchnorm/mul/ReadVariableOp0^layer_normalization_20/batchnorm/ReadVariableOp4^layer_normalization_20/batchnorm/mul/ReadVariableOp;^multi_head_attention_6/attention_output/add/ReadVariableOpE^multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_6/key/add/ReadVariableOp8^multi_head_attention_6/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/query/add/ReadVariableOp:^multi_head_attention_6/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/value/add/ReadVariableOp:^multi_head_attention_6/value/einsum/Einsum/ReadVariableOp1^positional_encoder_16/embedding/embedding_lookup-^sequential_6/dense_12/BiasAdd/ReadVariableOp/^sequential_6/dense_12/Tensordot/ReadVariableOp6^tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp5^tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp6^tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp5^tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp6^tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp5^tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp7^tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp6^tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp7^tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp6^tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp6^tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp5^tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2b
/layer_normalization_18/batchnorm/ReadVariableOp/layer_normalization_18/batchnorm/ReadVariableOp2j
3layer_normalization_18/batchnorm/mul/ReadVariableOp3layer_normalization_18/batchnorm/mul/ReadVariableOp2b
/layer_normalization_19/batchnorm/ReadVariableOp/layer_normalization_19/batchnorm/ReadVariableOp2j
3layer_normalization_19/batchnorm/mul/ReadVariableOp3layer_normalization_19/batchnorm/mul/ReadVariableOp2b
/layer_normalization_20/batchnorm/ReadVariableOp/layer_normalization_20/batchnorm/ReadVariableOp2j
3layer_normalization_20/batchnorm/mul/ReadVariableOp3layer_normalization_20/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_6/attention_output/add/ReadVariableOp:multi_head_attention_6/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_6/key/add/ReadVariableOp-multi_head_attention_6/key/add/ReadVariableOp2r
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/query/add/ReadVariableOp/multi_head_attention_6/query/add/ReadVariableOp2v
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/value/add/ReadVariableOp/multi_head_attention_6/value/add/ReadVariableOp2v
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2d
0positional_encoder_16/embedding/embedding_lookup0positional_encoder_16/embedding/embedding_lookup2\
,sequential_6/dense_12/BiasAdd/ReadVariableOp,sequential_6/dense_12/BiasAdd/ReadVariableOp2`
.sequential_6/dense_12/Tensordot/ReadVariableOp.sequential_6/dense_12/Tensordot/ReadVariableOp2n
5tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp5tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp2l
4tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp4tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp2n
5tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp5tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp2l
4tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp4tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp2n
5tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp5tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp2l
4tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp4tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp2p
6tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp6tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp2n
5tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp5tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp2p
6tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp6tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp2n
5tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp5tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp2n
5tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp5tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp2l
4tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp4tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_nameinputs: 

_output_shapes
:(
ѕ	
і
5__inference_tubelet_embedding_32_layer_call_fn_496119

videos%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@А
	unknown_4:	А
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallvideosunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_tubelet_embedding_32_layer_call_and_return_conditional_losses_494440t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_namevideos
ё
l
P__inference_average_pooling3d_33_layer_call_and_return_conditional_losses_496670

inputs
identityЊ
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Џ
h
L__inference_max_pooling3d_66_layer_call_and_return_conditional_losses_496650

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ё
n
B__inference_add_13_layer_call_and_return_conditional_losses_496549
inputs_0
inputs_1
identityW
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:€€€€€€€€€(АT
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€(А:€€€€€€€€€(А:V R
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/1
у
M
1__inference_max_pooling3d_64_layer_call_fn_496615

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *U
fPRN
L__inference_max_pooling3d_64_layer_call_and_return_conditional_losses_494190Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
у
M
1__inference_max_pooling3d_66_layer_call_fn_496645

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *U
fPRN
L__inference_max_pooling3d_66_layer_call_and_return_conditional_losses_494226Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
 
Ч
)__inference_dense_13_layer_call_fn_496600

inputs
unknown:	А
	unknown_0:
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_494702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
’
l
B__inference_add_12_layer_call_and_return_conditional_losses_494620

inputs
inputs_1
identityU
addAddV2inputsinputs_1*
T0*,
_output_shapes
:€€€€€€€€€(АT
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€(А:€€€€€€€€€(А:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
ё
l
P__inference_average_pooling3d_33_layer_call_and_return_conditional_losses_494250

inputs
identityЊ
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
њ3
Ч
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_496400	
query	
valueC
+query_einsum_einsum_readvariableop_resource:АА4
!query_add_readvariableop_resource:	АA
)key_einsum_einsum_readvariableop_resource:АА2
key_add_readvariableop_resource:	АC
+value_einsum_einsum_readvariableop_resource:АА4
!value_add_readvariableop_resource:	АN
6attention_output_einsum_einsum_readvariableop_resource:АА;
,attention_output_add_readvariableop_resource:	А
identity

identity_1ИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpФ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde{
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АР
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abdew
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes
:	А*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АФ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde{
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€(АП
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€((*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€((Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?П
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€((^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:∞
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€((*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=∆
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€((З
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€((Й
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€((¶
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationacbe,aecd->abcd™
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А*
equationabcd,cde->abeН
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аr

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€((Ў
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€(А:€€€€€€€€€(А: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namevalue
Ѕ
S
'__inference_add_12_layer_call_fn_496406
inputs_0
inputs_1
identityƒ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_494620e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€(А:€€€€€€€€€(А:V R
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/1
З
њ
(__inference_model_6_layer_call_fn_494776
input_17%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@А
	unknown_4:	А'
	unknown_5: 
	unknown_6: '
	unknown_7: @
	unknown_8:@(
	unknown_9:@А

unknown_10:	А

unknown_11

unknown_12:	(А

unknown_13:	А

unknown_14:	А"

unknown_15:АА

unknown_16:	А"

unknown_17:АА

unknown_18:	А"

unknown_19:АА

unknown_20:	А"

unknown_21:АА

unknown_22:	А

unknown_23:	А

unknown_24:	А

unknown_25:
АА

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8В *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_494709o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
3
_output_shapes!
:€€€€€€€€€

"
_user_specified_name
input_17: 

_output_shapes
:(
н

Ў
Q__inference_positional_encoder_16_layer_call_and_return_conditional_losses_494525
encoded_tokens
unknown4
!embedding_embedding_lookup_494518:	(А
identityИҐembedding/embedding_lookupћ
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_494518unknown*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/494518*
_output_shapes
:	(А*
dtype0і
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/494518*
_output_shapes
:	(АЙ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	(АГ
addAddV2encoded_tokens.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аc
NoOpNoOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€(А:(: 28
embedding/embedding_lookupembedding/embedding_lookup:\ X
,
_output_shapes
:€€€€€€€€€(А
(
_user_specified_nameencoded_tokens: 

_output_shapes
:(
Ж-
√
P__inference_tubelet_embedding_32_layer_call_and_return_conditional_losses_496155

videosF
(conv3d_96_conv3d_readvariableop_resource: 7
)conv3d_96_biasadd_readvariableop_resource: F
(conv3d_97_conv3d_readvariableop_resource: @7
)conv3d_97_biasadd_readvariableop_resource:@G
(conv3d_98_conv3d_readvariableop_resource:@А8
)conv3d_98_biasadd_readvariableop_resource:	А
identityИҐ conv3d_96/BiasAdd/ReadVariableOpҐconv3d_96/Conv3D/ReadVariableOpҐ conv3d_97/BiasAdd/ReadVariableOpҐconv3d_97/Conv3D/ReadVariableOpҐ conv3d_98/BiasAdd/ReadVariableOpҐconv3d_98/Conv3D/ReadVariableOpФ
conv3d_96/Conv3D/ReadVariableOpReadVariableOp(conv3d_96_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0≤
conv3d_96/Conv3DConv3Dvideos'conv3d_96/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
paddingSAME*
strides	
Ж
 conv3d_96/BiasAdd/ReadVariableOpReadVariableOp)conv3d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Я
conv3d_96/BiasAddBiasAddconv3d_96/Conv3D:output:0(conv3d_96/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 p
conv3d_96/ReluReluconv3d_96/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
 Ѕ
max_pooling3d_64/MaxPool3D	MaxPool3Dconv3d_96/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
ksize	
*
paddingVALID*
strides	
Ф
conv3d_97/Conv3D/ReadVariableOpReadVariableOp(conv3d_97_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0ѕ
conv3d_97/Conv3DConv3D#max_pooling3d_64/MaxPool3D:output:0'conv3d_97/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
paddingSAME*
strides	
Ж
 conv3d_97/BiasAdd/ReadVariableOpReadVariableOp)conv3d_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Я
conv3d_97/BiasAddBiasAddconv3d_97/Conv3D:output:0(conv3d_97/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@p
conv3d_97/ReluReluconv3d_97/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
@Ѕ
max_pooling3d_65/MaxPool3D	MaxPool3Dconv3d_97/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
ksize	
*
paddingVALID*
strides	
Х
conv3d_98/Conv3D/ReadVariableOpReadVariableOp(conv3d_98_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0–
conv3d_98/Conv3DConv3D#max_pooling3d_65/MaxPool3D:output:0'conv3d_98/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
paddingSAME*
strides	
З
 conv3d_98/BiasAdd/ReadVariableOpReadVariableOp)conv3d_98_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0†
conv3d_98/BiasAddBiasAddconv3d_98/Conv3D:output:0(conv3d_98/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
Аƒ
average_pooling3d_32/AvgPool3D	AvgPool3Dconv3d_98/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
ksize	
*
paddingVALID*
strides	
g
reshape_32/ShapeShape'average_pooling3d_32/AvgPool3D:output:0*
T0*
_output_shapes
:h
reshape_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
reshape_32/strided_sliceStridedSlicereshape_32/Shape:output:0'reshape_32/strided_slice/stack:output:0)reshape_32/strided_slice/stack_1:output:0)reshape_32/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
reshape_32/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€]
reshape_32/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Аї
reshape_32/Reshape/shapePack!reshape_32/strided_slice:output:0#reshape_32/Reshape/shape/1:output:0#reshape_32/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:†
reshape_32/ReshapeReshape'average_pooling3d_32/AvgPool3D:output:0!reshape_32/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аo
IdentityIdentityreshape_32/Reshape:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АХ
NoOpNoOp!^conv3d_96/BiasAdd/ReadVariableOp ^conv3d_96/Conv3D/ReadVariableOp!^conv3d_97/BiasAdd/ReadVariableOp ^conv3d_97/Conv3D/ReadVariableOp!^conv3d_98/BiasAdd/ReadVariableOp ^conv3d_98/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€
: : : : : : 2D
 conv3d_96/BiasAdd/ReadVariableOp conv3d_96/BiasAdd/ReadVariableOp2B
conv3d_96/Conv3D/ReadVariableOpconv3d_96/Conv3D/ReadVariableOp2D
 conv3d_97/BiasAdd/ReadVariableOp conv3d_97/BiasAdd/ReadVariableOp2B
conv3d_97/Conv3D/ReadVariableOpconv3d_97/Conv3D/ReadVariableOp2D
 conv3d_98/BiasAdd/ReadVariableOp conv3d_98/BiasAdd/ReadVariableOp2B
conv3d_98/Conv3D/ReadVariableOpconv3d_98/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_namevideos
©
Ґ
6__inference_positional_encoder_16_layer_call_fn_496230
encoded_tokens
unknown
	unknown_0:	(А
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallencoded_tokensunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Z
fURS
Q__inference_positional_encoder_16_layer_call_and_return_conditional_losses_494525t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€(А:(: 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:€€€€€€€€€(А
(
_user_specified_nameencoded_tokens: 

_output_shapes
:(
ю
У
R__inference_layer_normalization_19_layer_call_and_return_conditional_losses_494644

inputs4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(М
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ж
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0В
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АА
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
Ѕ
S
'__inference_add_13_layer_call_fn_496543
inputs_0
inputs_1
identityƒ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *K
fFRD
B__inference_add_13_layer_call_and_return_conditional_losses_494661e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€(А:€€€€€€€€€(А:V R
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/1
Џ
h
L__inference_max_pooling3d_65_layer_call_and_return_conditional_losses_496630

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ы
Q
5__inference_average_pooling3d_32_layer_call_fn_496635

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_average_pooling3d_32_layer_call_and_return_conditional_losses_494214Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Џ
h
L__inference_max_pooling3d_67_layer_call_and_return_conditional_losses_494238

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
б
v
J__inference_concatenate_13_layer_call_and_return_conditional_losses_496221
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :|
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€(А:€€€€€€€€€(А:V R
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/1
ё
Щ
)__inference_dense_12_layer_call_fn_496679

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_494298t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
њ3
Ч
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_494891	
query	
valueC
+query_einsum_einsum_readvariableop_resource:АА4
!query_add_readvariableop_resource:	АA
)key_einsum_einsum_readvariableop_resource:АА2
key_add_readvariableop_resource:	АC
+value_einsum_einsum_readvariableop_resource:АА4
!value_add_readvariableop_resource:	АN
6attention_output_einsum_einsum_readvariableop_resource:АА;
,attention_output_add_readvariableop_resource:	А
identity

identity_1ИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpФ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde{
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АР
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abdew
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes
:	А*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АФ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde{
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€(АП
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€((*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€((Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?П
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€((^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:∞
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€((*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=∆
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€((З
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€((Й
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€((¶
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationacbe,aecd->abcd™
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А*
equationabcd,cde->abeН
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аr

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€((Ў
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€(А:€€€€€€€€€(А: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namevalue
в
И
7__inference_multi_head_attention_6_layer_call_fn_496297	
query	
value
unknown:АА
	unknown_0:	А!
	unknown_1:АА
	unknown_2:	А!
	unknown_3:АА
	unknown_4:	А!
	unknown_5:АА
	unknown_6:	А
identity

identity_1ИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:€€€€€€€€€(А:€€€€€€€€€((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_494595t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аy

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:€€€€€€€€€((`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€(А:€€€€€€€€€(А: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namevalue
ы
Q
5__inference_average_pooling3d_33_layer_call_fn_496665

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_average_pooling3d_33_layer_call_and_return_conditional_losses_494250Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
АЌ
Ћ!
C__inference_model_6_layer_call_and_return_conditional_losses_495810

inputs[
=tubelet_embedding_32_conv3d_96_conv3d_readvariableop_resource: L
>tubelet_embedding_32_conv3d_96_biasadd_readvariableop_resource: [
=tubelet_embedding_32_conv3d_97_conv3d_readvariableop_resource: @L
>tubelet_embedding_32_conv3d_97_biasadd_readvariableop_resource:@\
=tubelet_embedding_32_conv3d_98_conv3d_readvariableop_resource:@АM
>tubelet_embedding_32_conv3d_98_biasadd_readvariableop_resource:	А[
=tubelet_embedding_33_conv3d_99_conv3d_readvariableop_resource: L
>tubelet_embedding_33_conv3d_99_biasadd_readvariableop_resource: \
>tubelet_embedding_33_conv3d_100_conv3d_readvariableop_resource: @M
?tubelet_embedding_33_conv3d_100_biasadd_readvariableop_resource:@]
>tubelet_embedding_33_conv3d_101_conv3d_readvariableop_resource:@АN
?tubelet_embedding_33_conv3d_101_biasadd_readvariableop_resource:	А 
positional_encoder_16_495673J
7positional_encoder_16_embedding_embedding_lookup_495675:	(АK
<layer_normalization_18_batchnorm_mul_readvariableop_resource:	АG
8layer_normalization_18_batchnorm_readvariableop_resource:	АZ
Bmulti_head_attention_6_query_einsum_einsum_readvariableop_resource:ААK
8multi_head_attention_6_query_add_readvariableop_resource:	АX
@multi_head_attention_6_key_einsum_einsum_readvariableop_resource:ААI
6multi_head_attention_6_key_add_readvariableop_resource:	АZ
Bmulti_head_attention_6_value_einsum_einsum_readvariableop_resource:ААK
8multi_head_attention_6_value_add_readvariableop_resource:	Аe
Mmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource:ААR
Cmulti_head_attention_6_attention_output_add_readvariableop_resource:	АK
<layer_normalization_19_batchnorm_mul_readvariableop_resource:	АG
8layer_normalization_19_batchnorm_readvariableop_resource:	АK
7sequential_6_dense_12_tensordot_readvariableop_resource:
ААD
5sequential_6_dense_12_biasadd_readvariableop_resource:	АK
<layer_normalization_20_batchnorm_mul_readvariableop_resource:	АG
8layer_normalization_20_batchnorm_readvariableop_resource:	А:
'dense_13_matmul_readvariableop_resource:	А6
(dense_13_biasadd_readvariableop_resource:
identityИҐdense_13/BiasAdd/ReadVariableOpҐdense_13/MatMul/ReadVariableOpҐ/layer_normalization_18/batchnorm/ReadVariableOpҐ3layer_normalization_18/batchnorm/mul/ReadVariableOpҐ/layer_normalization_19/batchnorm/ReadVariableOpҐ3layer_normalization_19/batchnorm/mul/ReadVariableOpҐ/layer_normalization_20/batchnorm/ReadVariableOpҐ3layer_normalization_20/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_6/attention_output/add/ReadVariableOpҐDmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_6/key/add/ReadVariableOpҐ7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_6/query/add/ReadVariableOpҐ9multi_head_attention_6/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_6/value/add/ReadVariableOpҐ9multi_head_attention_6/value/einsum/Einsum/ReadVariableOpҐ0positional_encoder_16/embedding/embedding_lookupҐ,sequential_6/dense_12/BiasAdd/ReadVariableOpҐ.sequential_6/dense_12/Tensordot/ReadVariableOpҐ5tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOpҐ4tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOpҐ5tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOpҐ4tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOpҐ5tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOpҐ4tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOpҐ6tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOpҐ5tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOpҐ6tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOpҐ5tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOpҐ5tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOpҐ4tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOpМ
/tf.__operators__.getitem_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               а
)tf.__operators__.getitem_37/strided_sliceStridedSliceinputs8tf.__operators__.getitem_37/strided_slice/stack:output:0:tf.__operators__.getitem_37/strided_slice/stack_1:output:0:tf.__operators__.getitem_37/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_maskМ
/tf.__operators__.getitem_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               а
)tf.__operators__.getitem_36/strided_sliceStridedSliceinputs8tf.__operators__.getitem_36/strided_slice/stack:output:0:tf.__operators__.getitem_36/strided_slice/stack_1:output:0:tf.__operators__.getitem_36/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_maskЊ
4tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_32_conv3d_96_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0И
%tubelet_embedding_32/conv3d_96/Conv3DConv3D2tf.__operators__.getitem_36/strided_slice:output:0<tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
paddingSAME*
strides	
∞
5tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_32_conv3d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ё
&tubelet_embedding_32/conv3d_96/BiasAddBiasAdd.tubelet_embedding_32/conv3d_96/Conv3D:output:0=tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 Ъ
#tubelet_embedding_32/conv3d_96/ReluRelu/tubelet_embedding_32/conv3d_96/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
 л
/tubelet_embedding_32/max_pooling3d_64/MaxPool3D	MaxPool3D1tubelet_embedding_32/conv3d_96/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
ksize	
*
paddingVALID*
strides	
Њ
4tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_32_conv3d_97_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0О
%tubelet_embedding_32/conv3d_97/Conv3DConv3D8tubelet_embedding_32/max_pooling3d_64/MaxPool3D:output:0<tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
paddingSAME*
strides	
∞
5tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_32_conv3d_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ё
&tubelet_embedding_32/conv3d_97/BiasAddBiasAdd.tubelet_embedding_32/conv3d_97/Conv3D:output:0=tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@Ъ
#tubelet_embedding_32/conv3d_97/ReluRelu/tubelet_embedding_32/conv3d_97/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
@л
/tubelet_embedding_32/max_pooling3d_65/MaxPool3D	MaxPool3D1tubelet_embedding_32/conv3d_97/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
ksize	
*
paddingVALID*
strides	
њ
4tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_32_conv3d_98_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0П
%tubelet_embedding_32/conv3d_98/Conv3DConv3D8tubelet_embedding_32/max_pooling3d_65/MaxPool3D:output:0<tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
paddingSAME*
strides	
±
5tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_32_conv3d_98_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0я
&tubelet_embedding_32/conv3d_98/BiasAddBiasAdd.tubelet_embedding_32/conv3d_98/Conv3D:output:0=tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
Ао
3tubelet_embedding_32/average_pooling3d_32/AvgPool3D	AvgPool3D/tubelet_embedding_32/conv3d_98/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
ksize	
*
paddingVALID*
strides	
С
%tubelet_embedding_32/reshape_32/ShapeShape<tubelet_embedding_32/average_pooling3d_32/AvgPool3D:output:0*
T0*
_output_shapes
:}
3tubelet_embedding_32/reshape_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tubelet_embedding_32/reshape_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tubelet_embedding_32/reshape_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-tubelet_embedding_32/reshape_32/strided_sliceStridedSlice.tubelet_embedding_32/reshape_32/Shape:output:0<tubelet_embedding_32/reshape_32/strided_slice/stack:output:0>tubelet_embedding_32/reshape_32/strided_slice/stack_1:output:0>tubelet_embedding_32/reshape_32/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/tubelet_embedding_32/reshape_32/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€r
/tubelet_embedding_32/reshape_32/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АП
-tubelet_embedding_32/reshape_32/Reshape/shapePack6tubelet_embedding_32/reshape_32/strided_slice:output:08tubelet_embedding_32/reshape_32/Reshape/shape/1:output:08tubelet_embedding_32/reshape_32/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:я
'tubelet_embedding_32/reshape_32/ReshapeReshape<tubelet_embedding_32/average_pooling3d_32/AvgPool3D:output:06tubelet_embedding_32/reshape_32/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АЊ
4tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_33_conv3d_99_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0И
%tubelet_embedding_33/conv3d_99/Conv3DConv3D2tf.__operators__.getitem_37/strided_slice:output:0<tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
paddingSAME*
strides	
∞
5tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_33_conv3d_99_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ё
&tubelet_embedding_33/conv3d_99/BiasAddBiasAdd.tubelet_embedding_33/conv3d_99/Conv3D:output:0=tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 Ъ
#tubelet_embedding_33/conv3d_99/ReluRelu/tubelet_embedding_33/conv3d_99/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
 л
/tubelet_embedding_33/max_pooling3d_66/MaxPool3D	MaxPool3D1tubelet_embedding_33/conv3d_99/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
ksize	
*
paddingVALID*
strides	
ј
5tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOpReadVariableOp>tubelet_embedding_33_conv3d_100_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0Р
&tubelet_embedding_33/conv3d_100/Conv3DConv3D8tubelet_embedding_33/max_pooling3d_66/MaxPool3D:output:0=tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
paddingSAME*
strides	
≤
6tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOpReadVariableOp?tubelet_embedding_33_conv3d_100_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0б
'tubelet_embedding_33/conv3d_100/BiasAddBiasAdd/tubelet_embedding_33/conv3d_100/Conv3D:output:0>tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@Ь
$tubelet_embedding_33/conv3d_100/ReluRelu0tubelet_embedding_33/conv3d_100/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
@м
/tubelet_embedding_33/max_pooling3d_67/MaxPool3D	MaxPool3D2tubelet_embedding_33/conv3d_100/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
ksize	
*
paddingVALID*
strides	
Ѕ
5tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOpReadVariableOp>tubelet_embedding_33_conv3d_101_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0С
&tubelet_embedding_33/conv3d_101/Conv3DConv3D8tubelet_embedding_33/max_pooling3d_67/MaxPool3D:output:0=tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
paddingSAME*
strides	
≥
6tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOpReadVariableOp?tubelet_embedding_33_conv3d_101_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0в
'tubelet_embedding_33/conv3d_101/BiasAddBiasAdd/tubelet_embedding_33/conv3d_101/Conv3D:output:0>tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
Ап
3tubelet_embedding_33/average_pooling3d_33/AvgPool3D	AvgPool3D0tubelet_embedding_33/conv3d_101/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
ksize	
*
paddingVALID*
strides	
С
%tubelet_embedding_33/reshape_33/ShapeShape<tubelet_embedding_33/average_pooling3d_33/AvgPool3D:output:0*
T0*
_output_shapes
:}
3tubelet_embedding_33/reshape_33/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tubelet_embedding_33/reshape_33/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tubelet_embedding_33/reshape_33/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-tubelet_embedding_33/reshape_33/strided_sliceStridedSlice.tubelet_embedding_33/reshape_33/Shape:output:0<tubelet_embedding_33/reshape_33/strided_slice/stack:output:0>tubelet_embedding_33/reshape_33/strided_slice/stack_1:output:0>tubelet_embedding_33/reshape_33/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/tubelet_embedding_33/reshape_33/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€r
/tubelet_embedding_33/reshape_33/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АП
-tubelet_embedding_33/reshape_33/Reshape/shapePack6tubelet_embedding_33/reshape_33/strided_slice:output:08tubelet_embedding_33/reshape_33/Reshape/shape/1:output:08tubelet_embedding_33/reshape_33/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:я
'tubelet_embedding_33/reshape_33/ReshapeReshape<tubelet_embedding_33/average_pooling3d_33/AvgPool3D:output:06tubelet_embedding_33/reshape_33/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А\
concatenate_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :к
concatenate_13/concatConcatV20tubelet_embedding_32/reshape_32/Reshape:output:00tubelet_embedding_33/reshape_33/Reshape:output:0#concatenate_13/concat/axis:output:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А£
0positional_encoder_16/embedding/embedding_lookupResourceGather7positional_encoder_16_embedding_embedding_lookup_495675positional_encoder_16_495673*
Tindices0*J
_class@
><loc:@positional_encoder_16/embedding/embedding_lookup/495675*
_output_shapes
:	(А*
dtype0ц
9positional_encoder_16/embedding/embedding_lookup/IdentityIdentity9positional_encoder_16/embedding/embedding_lookup:output:0*
T0*J
_class@
><loc:@positional_encoder_16/embedding/embedding_lookup/495675*
_output_shapes
:	(Аµ
;positional_encoder_16/embedding/embedding_lookup/Identity_1IdentityBpositional_encoder_16/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	(Ањ
positional_encoder_16/addAddV2concatenate_13/concat:output:0Dpositional_encoder_16/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А
5layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:—
#layer_normalization_18/moments/meanMeanpositional_encoder_16/add:z:0>layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(Я
+layer_normalization_18/moments/StopGradientStopGradient,layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(—
0layer_normalization_18/moments/SquaredDifferenceSquaredDifferencepositional_encoder_16/add:z:04layer_normalization_18/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АГ
9layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:р
'layer_normalization_18/moments/varianceMean4layer_normalization_18/moments/SquaredDifference:z:0Blayer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(k
&layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5∆
$layer_normalization_18/batchnorm/addAddV20layer_normalization_18/moments/variance:output:0/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(П
&layer_normalization_18/batchnorm/RsqrtRsqrt(layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(≠
3layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ћ
$layer_normalization_18/batchnorm/mulMul*layer_normalization_18/batchnorm/Rsqrt:y:0;layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(А≠
&layer_normalization_18/batchnorm/mul_1Mulpositional_encoder_16/add:z:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_18/batchnorm/mul_2Mul,layer_normalization_18/moments/mean:output:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А•
/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0«
$layer_normalization_18/batchnorm/subSub7layer_normalization_18/batchnorm/ReadVariableOp:value:0*layer_normalization_18/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_18/batchnorm/add_1AddV2*layer_normalization_18/batchnorm/mul_1:z:0(layer_normalization_18/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А¬
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0Д
*multi_head_attention_6/query/einsum/EinsumEinsum*layer_normalization_18/batchnorm/add_1:z:0Amulti_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde©
/multi_head_attention_6/query/add/ReadVariableOpReadVariableOp8multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes
:	А*
dtype0“
 multi_head_attention_6/query/addAddV23multi_head_attention_6/query/einsum/Einsum:output:07multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АЊ
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_6_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0А
(multi_head_attention_6/key/einsum/EinsumEinsum*layer_normalization_18/batchnorm/add_1:z:0?multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde•
-multi_head_attention_6/key/add/ReadVariableOpReadVariableOp6multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes
:	А*
dtype0ћ
multi_head_attention_6/key/addAddV21multi_head_attention_6/key/einsum/Einsum:output:05multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(А¬
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0Д
*multi_head_attention_6/value/einsum/EinsumEinsum*layer_normalization_18/batchnorm/add_1:z:0Amulti_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde©
/multi_head_attention_6/value/add/ReadVariableOpReadVariableOp8multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes
:	А*
dtype0“
 multi_head_attention_6/value/addAddV23multi_head_attention_6/value/einsum/Einsum:output:07multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(Аa
multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=©
multi_head_attention_6/MulMul$multi_head_attention_6/query/add:z:0%multi_head_attention_6/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€(А‘
$multi_head_attention_6/einsum/EinsumEinsum"multi_head_attention_6/key/add:z:0multi_head_attention_6/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€((*
equationaecd,abcd->acbeЪ
&multi_head_attention_6/softmax/SoftmaxSoftmax-multi_head_attention_6/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€((Я
'multi_head_attention_6/dropout/IdentityIdentity0multi_head_attention_6/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€((л
&multi_head_attention_6/einsum_1/EinsumEinsum0multi_head_attention_6/dropout/Identity:output:0$multi_head_attention_6/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationacbe,aecd->abcdЎ
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0Ы
5multi_head_attention_6/attention_output/einsum/EinsumEinsum/multi_head_attention_6/einsum_1/Einsum:output:0Lmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А*
equationabcd,cde->abeї
:multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype0п
+multi_head_attention_6/attention_output/addAddV2>multi_head_attention_6/attention_output/einsum/Einsum:output:0Bmulti_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(АЪ

add_12/addAddV2/multi_head_attention_6/attention_output/add:z:0positional_encoder_16/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А
5layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
#layer_normalization_19/moments/meanMeanadd_12/add:z:0>layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(Я
+layer_normalization_19/moments/StopGradientStopGradient,layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(¬
0layer_normalization_19/moments/SquaredDifferenceSquaredDifferenceadd_12/add:z:04layer_normalization_19/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АГ
9layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:р
'layer_normalization_19/moments/varianceMean4layer_normalization_19/moments/SquaredDifference:z:0Blayer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(k
&layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5∆
$layer_normalization_19/batchnorm/addAddV20layer_normalization_19/moments/variance:output:0/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(П
&layer_normalization_19/batchnorm/RsqrtRsqrt(layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(≠
3layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ћ
$layer_normalization_19/batchnorm/mulMul*layer_normalization_19/batchnorm/Rsqrt:y:0;layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(АЮ
&layer_normalization_19/batchnorm/mul_1Muladd_12/add:z:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_19/batchnorm/mul_2Mul,layer_normalization_19/moments/mean:output:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А•
/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0«
$layer_normalization_19/batchnorm/subSub7layer_normalization_19/batchnorm/ReadVariableOp:value:0*layer_normalization_19/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_19/batchnorm/add_1AddV2*layer_normalization_19/batchnorm/mul_1:z:0(layer_normalization_19/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А®
.sequential_6/dense_12/Tensordot/ReadVariableOpReadVariableOp7sequential_6_dense_12_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype0n
$sequential_6/dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_6/dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_6/dense_12/Tensordot/ShapeShape*layer_normalization_19/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_6/dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : У
(sequential_6/dense_12/Tensordot/GatherV2GatherV2.sequential_6/dense_12/Tensordot/Shape:output:0-sequential_6/dense_12/Tensordot/free:output:06sequential_6/dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_6/dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
*sequential_6/dense_12/Tensordot/GatherV2_1GatherV2.sequential_6/dense_12/Tensordot/Shape:output:0-sequential_6/dense_12/Tensordot/axes:output:08sequential_6/dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_6/dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ∞
$sequential_6/dense_12/Tensordot/ProdProd1sequential_6/dense_12/Tensordot/GatherV2:output:0.sequential_6/dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_6/dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ґ
&sequential_6/dense_12/Tensordot/Prod_1Prod3sequential_6/dense_12/Tensordot/GatherV2_1:output:00sequential_6/dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_6/dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
&sequential_6/dense_12/Tensordot/concatConcatV2-sequential_6/dense_12/Tensordot/free:output:0-sequential_6/dense_12/Tensordot/axes:output:04sequential_6/dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ї
%sequential_6/dense_12/Tensordot/stackPack-sequential_6/dense_12/Tensordot/Prod:output:0/sequential_6/dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
)sequential_6/dense_12/Tensordot/transpose	Transpose*layer_normalization_19/batchnorm/add_1:z:0/sequential_6/dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аћ
'sequential_6/dense_12/Tensordot/ReshapeReshape-sequential_6/dense_12/Tensordot/transpose:y:0.sequential_6/dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Ќ
&sequential_6/dense_12/Tensordot/MatMulMatMul0sequential_6/dense_12/Tensordot/Reshape:output:06sequential_6/dense_12/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аr
'sequential_6/dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аo
-sequential_6/dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : €
(sequential_6/dense_12/Tensordot/concat_1ConcatV21sequential_6/dense_12/Tensordot/GatherV2:output:00sequential_6/dense_12/Tensordot/Const_2:output:06sequential_6/dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:∆
sequential_6/dense_12/TensordotReshape0sequential_6/dense_12/Tensordot/MatMul:product:01sequential_6/dense_12/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АЯ
,sequential_6/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0њ
sequential_6/dense_12/BiasAddBiasAdd(sequential_6/dense_12/Tensordot:output:04sequential_6/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аe
 sequential_6/dense_12/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ѓ
sequential_6/dense_12/Gelu/mulMul)sequential_6/dense_12/Gelu/mul/x:output:0&sequential_6/dense_12/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аf
!sequential_6/dense_12/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?Є
"sequential_6/dense_12/Gelu/truedivRealDiv&sequential_6/dense_12/BiasAdd:output:0*sequential_6/dense_12/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АД
sequential_6/dense_12/Gelu/ErfErf&sequential_6/dense_12/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аe
 sequential_6/dense_12/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≠
sequential_6/dense_12/Gelu/addAddV2)sequential_6/dense_12/Gelu/add/x:output:0"sequential_6/dense_12/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€(А¶
 sequential_6/dense_12/Gelu/mul_1Mul"sequential_6/dense_12/Gelu/mul:z:0"sequential_6/dense_12/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АА

add_13/addAddV2$sequential_6/dense_12/Gelu/mul_1:z:0add_12/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А
5layer_normalization_20/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
#layer_normalization_20/moments/meanMeanadd_13/add:z:0>layer_normalization_20/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(Я
+layer_normalization_20/moments/StopGradientStopGradient,layer_normalization_20/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(¬
0layer_normalization_20/moments/SquaredDifferenceSquaredDifferenceadd_13/add:z:04layer_normalization_20/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АГ
9layer_normalization_20/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:р
'layer_normalization_20/moments/varianceMean4layer_normalization_20/moments/SquaredDifference:z:0Blayer_normalization_20/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(k
&layer_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5∆
$layer_normalization_20/batchnorm/addAddV20layer_normalization_20/moments/variance:output:0/layer_normalization_20/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(П
&layer_normalization_20/batchnorm/RsqrtRsqrt(layer_normalization_20/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(≠
3layer_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ћ
$layer_normalization_20/batchnorm/mulMul*layer_normalization_20/batchnorm/Rsqrt:y:0;layer_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(АЮ
&layer_normalization_20/batchnorm/mul_1Muladd_13/add:z:0(layer_normalization_20/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_20/batchnorm/mul_2Mul,layer_normalization_20/moments/mean:output:0(layer_normalization_20/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А•
/layer_normalization_20/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0«
$layer_normalization_20/batchnorm/subSub7layer_normalization_20/batchnorm/ReadVariableOp:value:0*layer_normalization_20/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АЉ
&layer_normalization_20/batchnorm/add_1AddV2*layer_normalization_20/batchnorm/mul_1:z:0(layer_normalization_20/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аs
1global_average_pooling1d_6/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¬
global_average_pooling1d_6/MeanMean*layer_normalization_20/batchnorm/add_1:z:0:global_average_pooling1d_6/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЗ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Э
dense_13/MatMulMatMul(global_average_pooling1d_6/Mean:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_13/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Љ
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp0^layer_normalization_18/batchnorm/ReadVariableOp4^layer_normalization_18/batchnorm/mul/ReadVariableOp0^layer_normalization_19/batchnorm/ReadVariableOp4^layer_normalization_19/batchnorm/mul/ReadVariableOp0^layer_normalization_20/batchnorm/ReadVariableOp4^layer_normalization_20/batchnorm/mul/ReadVariableOp;^multi_head_attention_6/attention_output/add/ReadVariableOpE^multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_6/key/add/ReadVariableOp8^multi_head_attention_6/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/query/add/ReadVariableOp:^multi_head_attention_6/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/value/add/ReadVariableOp:^multi_head_attention_6/value/einsum/Einsum/ReadVariableOp1^positional_encoder_16/embedding/embedding_lookup-^sequential_6/dense_12/BiasAdd/ReadVariableOp/^sequential_6/dense_12/Tensordot/ReadVariableOp6^tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp5^tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp6^tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp5^tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp6^tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp5^tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp7^tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp6^tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp7^tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp6^tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp6^tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp5^tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2b
/layer_normalization_18/batchnorm/ReadVariableOp/layer_normalization_18/batchnorm/ReadVariableOp2j
3layer_normalization_18/batchnorm/mul/ReadVariableOp3layer_normalization_18/batchnorm/mul/ReadVariableOp2b
/layer_normalization_19/batchnorm/ReadVariableOp/layer_normalization_19/batchnorm/ReadVariableOp2j
3layer_normalization_19/batchnorm/mul/ReadVariableOp3layer_normalization_19/batchnorm/mul/ReadVariableOp2b
/layer_normalization_20/batchnorm/ReadVariableOp/layer_normalization_20/batchnorm/ReadVariableOp2j
3layer_normalization_20/batchnorm/mul/ReadVariableOp3layer_normalization_20/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_6/attention_output/add/ReadVariableOp:multi_head_attention_6/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_6/key/add/ReadVariableOp-multi_head_attention_6/key/add/ReadVariableOp2r
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/query/add/ReadVariableOp/multi_head_attention_6/query/add/ReadVariableOp2v
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/value/add/ReadVariableOp/multi_head_attention_6/value/add/ReadVariableOp2v
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2d
0positional_encoder_16/embedding/embedding_lookup0positional_encoder_16/embedding/embedding_lookup2\
,sequential_6/dense_12/BiasAdd/ReadVariableOp,sequential_6/dense_12/BiasAdd/ReadVariableOp2`
.sequential_6/dense_12/Tensordot/ReadVariableOp.sequential_6/dense_12/Tensordot/ReadVariableOp2n
5tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp5tubelet_embedding_32/conv3d_96/BiasAdd/ReadVariableOp2l
4tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp4tubelet_embedding_32/conv3d_96/Conv3D/ReadVariableOp2n
5tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp5tubelet_embedding_32/conv3d_97/BiasAdd/ReadVariableOp2l
4tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp4tubelet_embedding_32/conv3d_97/Conv3D/ReadVariableOp2n
5tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp5tubelet_embedding_32/conv3d_98/BiasAdd/ReadVariableOp2l
4tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp4tubelet_embedding_32/conv3d_98/Conv3D/ReadVariableOp2p
6tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp6tubelet_embedding_33/conv3d_100/BiasAdd/ReadVariableOp2n
5tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp5tubelet_embedding_33/conv3d_100/Conv3D/ReadVariableOp2p
6tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp6tubelet_embedding_33/conv3d_101/BiasAdd/ReadVariableOp2n
5tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp5tubelet_embedding_33/conv3d_101/Conv3D/ReadVariableOp2n
5tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp5tubelet_embedding_33/conv3d_99/BiasAdd/ReadVariableOp2l
4tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp4tubelet_embedding_33/conv3d_99/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_nameinputs: 

_output_shapes
:(
«(
¶
H__inference_sequential_6_layer_call_and_return_conditional_losses_496499

inputs>
*dense_12_tensordot_readvariableop_resource:
АА7
(dense_12_biasadd_readvariableop_resource:	А
identityИҐdense_12/BiasAdd/ReadVariableOpҐ!dense_12/Tensordot/ReadVariableOpО
!dense_12/Tensordot/ReadVariableOpReadVariableOp*dense_12_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype0a
dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_12/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : я
dense_12/Tensordot/GatherV2GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/free:output:0)dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
dense_12/Tensordot/GatherV2_1GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/axes:output:0+dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Й
dense_12/Tensordot/ProdProd$dense_12/Tensordot/GatherV2:output:0!dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
dense_12/Tensordot/Prod_1Prod&dense_12/Tensordot/GatherV2_1:output:0#dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ј
dense_12/Tensordot/concatConcatV2 dense_12/Tensordot/free:output:0 dense_12/Tensordot/axes:output:0'dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
dense_12/Tensordot/stackPack dense_12/Tensordot/Prod:output:0"dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:М
dense_12/Tensordot/transpose	Transposeinputs"dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А•
dense_12/Tensordot/ReshapeReshape dense_12/Tensordot/transpose:y:0!dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€¶
dense_12/Tensordot/MatMulMatMul#dense_12/Tensordot/Reshape:output:0)dense_12/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аe
dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аb
 dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ћ
dense_12/Tensordot/concat_1ConcatV2$dense_12/Tensordot/GatherV2:output:0#dense_12/Tensordot/Const_2:output:0)dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Я
dense_12/TensordotReshape#dense_12/Tensordot/MatMul:product:0$dense_12/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АЕ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ш
dense_12/BiasAddBiasAdddense_12/Tensordot:output:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(АX
dense_12/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?И
dense_12/Gelu/mulMuldense_12/Gelu/mul/x:output:0dense_12/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АY
dense_12/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?С
dense_12/Gelu/truedivRealDivdense_12/BiasAdd:output:0dense_12/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аj
dense_12/Gelu/ErfErfdense_12/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АX
dense_12/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ж
dense_12/Gelu/addAddV2dense_12/Gelu/add/x:output:0dense_12/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€(А
dense_12/Gelu/mul_1Muldense_12/Gelu/mul:z:0dense_12/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аk
IdentityIdentitydense_12/Gelu/mul_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АМ
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp"^dense_12/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2F
!dense_12/Tensordot/ReadVariableOp!dense_12/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
Б
љ
(__inference_model_6_layer_call_fn_495527

inputs%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@А
	unknown_4:	А'
	unknown_5: 
	unknown_6: '
	unknown_7: @
	unknown_8:@(
	unknown_9:@А

unknown_10:	А

unknown_11

unknown_12:	(А

unknown_13:	А

unknown_14:	А"

unknown_15:АА

unknown_16:	А"

unknown_17:АА

unknown_18:	А"

unknown_19:АА

unknown_20:	А"

unknown_21:АА

unknown_22:	А

unknown_23:	А

unknown_24:	А

unknown_25:
АА

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8В *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_494709o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_nameinputs: 

_output_shapes
:(
∞-
Ћ
P__inference_tubelet_embedding_33_layer_call_and_return_conditional_losses_496208

videosF
(conv3d_99_conv3d_readvariableop_resource: 7
)conv3d_99_biasadd_readvariableop_resource: G
)conv3d_100_conv3d_readvariableop_resource: @8
*conv3d_100_biasadd_readvariableop_resource:@H
)conv3d_101_conv3d_readvariableop_resource:@А9
*conv3d_101_biasadd_readvariableop_resource:	А
identityИҐ!conv3d_100/BiasAdd/ReadVariableOpҐ conv3d_100/Conv3D/ReadVariableOpҐ!conv3d_101/BiasAdd/ReadVariableOpҐ conv3d_101/Conv3D/ReadVariableOpҐ conv3d_99/BiasAdd/ReadVariableOpҐconv3d_99/Conv3D/ReadVariableOpФ
conv3d_99/Conv3D/ReadVariableOpReadVariableOp(conv3d_99_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0≤
conv3d_99/Conv3DConv3Dvideos'conv3d_99/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
paddingSAME*
strides	
Ж
 conv3d_99/BiasAdd/ReadVariableOpReadVariableOp)conv3d_99_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Я
conv3d_99/BiasAddBiasAddconv3d_99/Conv3D:output:0(conv3d_99/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 p
conv3d_99/ReluReluconv3d_99/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
 Ѕ
max_pooling3d_66/MaxPool3D	MaxPool3Dconv3d_99/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
ksize	
*
paddingVALID*
strides	
Ц
 conv3d_100/Conv3D/ReadVariableOpReadVariableOp)conv3d_100_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0—
conv3d_100/Conv3DConv3D#max_pooling3d_66/MaxPool3D:output:0(conv3d_100/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
paddingSAME*
strides	
И
!conv3d_100/BiasAdd/ReadVariableOpReadVariableOp*conv3d_100_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ґ
conv3d_100/BiasAddBiasAddconv3d_100/Conv3D:output:0)conv3d_100/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@r
conv3d_100/ReluReluconv3d_100/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
@¬
max_pooling3d_67/MaxPool3D	MaxPool3Dconv3d_100/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
ksize	
*
paddingVALID*
strides	
Ч
 conv3d_101/Conv3D/ReadVariableOpReadVariableOp)conv3d_101_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0“
conv3d_101/Conv3DConv3D#max_pooling3d_67/MaxPool3D:output:0(conv3d_101/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
paddingSAME*
strides	
Й
!conv3d_101/BiasAdd/ReadVariableOpReadVariableOp*conv3d_101_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0£
conv3d_101/BiasAddBiasAddconv3d_101/Conv3D:output:0)conv3d_101/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А≈
average_pooling3d_33/AvgPool3D	AvgPool3Dconv3d_101/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
ksize	
*
paddingVALID*
strides	
g
reshape_33/ShapeShape'average_pooling3d_33/AvgPool3D:output:0*
T0*
_output_shapes
:h
reshape_33/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_33/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_33/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
reshape_33/strided_sliceStridedSlicereshape_33/Shape:output:0'reshape_33/strided_slice/stack:output:0)reshape_33/strided_slice/stack_1:output:0)reshape_33/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
reshape_33/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€]
reshape_33/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Аї
reshape_33/Reshape/shapePack!reshape_33/strided_slice:output:0#reshape_33/Reshape/shape/1:output:0#reshape_33/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:†
reshape_33/ReshapeReshape'average_pooling3d_33/AvgPool3D:output:0!reshape_33/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аo
IdentityIdentityreshape_33/Reshape:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АЩ
NoOpNoOp"^conv3d_100/BiasAdd/ReadVariableOp!^conv3d_100/Conv3D/ReadVariableOp"^conv3d_101/BiasAdd/ReadVariableOp!^conv3d_101/Conv3D/ReadVariableOp!^conv3d_99/BiasAdd/ReadVariableOp ^conv3d_99/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€
: : : : : : 2F
!conv3d_100/BiasAdd/ReadVariableOp!conv3d_100/BiasAdd/ReadVariableOp2D
 conv3d_100/Conv3D/ReadVariableOp conv3d_100/Conv3D/ReadVariableOp2F
!conv3d_101/BiasAdd/ReadVariableOp!conv3d_101/BiasAdd/ReadVariableOp2D
 conv3d_101/Conv3D/ReadVariableOp conv3d_101/Conv3D/ReadVariableOp2D
 conv3d_99/BiasAdd/ReadVariableOp conv3d_99/BiasAdd/ReadVariableOp2B
conv3d_99/Conv3D/ReadVariableOpconv3d_99/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_namevideos
Ћ	
ц
D__inference_dense_13_layer_call_and_return_conditional_losses_496610

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
«(
¶
H__inference_sequential_6_layer_call_and_return_conditional_losses_496537

inputs>
*dense_12_tensordot_readvariableop_resource:
АА7
(dense_12_biasadd_readvariableop_resource:	А
identityИҐdense_12/BiasAdd/ReadVariableOpҐ!dense_12/Tensordot/ReadVariableOpО
!dense_12/Tensordot/ReadVariableOpReadVariableOp*dense_12_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype0a
dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_12/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : я
dense_12/Tensordot/GatherV2GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/free:output:0)dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
dense_12/Tensordot/GatherV2_1GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/axes:output:0+dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Й
dense_12/Tensordot/ProdProd$dense_12/Tensordot/GatherV2:output:0!dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
dense_12/Tensordot/Prod_1Prod&dense_12/Tensordot/GatherV2_1:output:0#dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ј
dense_12/Tensordot/concatConcatV2 dense_12/Tensordot/free:output:0 dense_12/Tensordot/axes:output:0'dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
dense_12/Tensordot/stackPack dense_12/Tensordot/Prod:output:0"dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:М
dense_12/Tensordot/transpose	Transposeinputs"dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А•
dense_12/Tensordot/ReshapeReshape dense_12/Tensordot/transpose:y:0!dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€¶
dense_12/Tensordot/MatMulMatMul#dense_12/Tensordot/Reshape:output:0)dense_12/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аe
dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Аb
 dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ћ
dense_12/Tensordot/concat_1ConcatV2$dense_12/Tensordot/GatherV2:output:0#dense_12/Tensordot/Const_2:output:0)dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Я
dense_12/TensordotReshape#dense_12/Tensordot/MatMul:product:0$dense_12/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АЕ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ш
dense_12/BiasAddBiasAdddense_12/Tensordot:output:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(АX
dense_12/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?И
dense_12/Gelu/mulMuldense_12/Gelu/mul/x:output:0dense_12/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АY
dense_12/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?С
dense_12/Gelu/truedivRealDivdense_12/BiasAdd:output:0dense_12/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аj
dense_12/Gelu/ErfErfdense_12/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АX
dense_12/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ж
dense_12/Gelu/addAddV2dense_12/Gelu/add/x:output:0dense_12/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€(А
dense_12/Gelu/mul_1Muldense_12/Gelu/mul:z:0dense_12/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аk
IdentityIdentitydense_12/Gelu/mul_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АМ
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp"^dense_12/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2F
!dense_12/Tensordot/ReadVariableOp!dense_12/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
л[
€
C__inference_model_6_layer_call_and_return_conditional_losses_495452
input_179
tubelet_embedding_32_495373: )
tubelet_embedding_32_495375: 9
tubelet_embedding_32_495377: @)
tubelet_embedding_32_495379:@:
tubelet_embedding_32_495381:@А*
tubelet_embedding_32_495383:	А9
tubelet_embedding_33_495386: )
tubelet_embedding_33_495388: 9
tubelet_embedding_33_495390: @)
tubelet_embedding_33_495392:@:
tubelet_embedding_33_495394:@А*
tubelet_embedding_33_495396:	А 
positional_encoder_16_495400/
positional_encoder_16_495402:	(А,
layer_normalization_18_495405:	А,
layer_normalization_18_495407:	А5
multi_head_attention_6_495410:АА0
multi_head_attention_6_495412:	А5
multi_head_attention_6_495414:АА0
multi_head_attention_6_495416:	А5
multi_head_attention_6_495418:АА0
multi_head_attention_6_495420:	А5
multi_head_attention_6_495422:АА,
multi_head_attention_6_495424:	А,
layer_normalization_19_495429:	А,
layer_normalization_19_495431:	А'
sequential_6_495434:
АА"
sequential_6_495436:	А,
layer_normalization_20_495440:	А,
layer_normalization_20_495442:	А"
dense_13_495446:	А
dense_13_495448:
identityИҐ dense_13/StatefulPartitionedCallҐ.layer_normalization_18/StatefulPartitionedCallҐ.layer_normalization_19/StatefulPartitionedCallҐ.layer_normalization_20/StatefulPartitionedCallҐ.multi_head_attention_6/StatefulPartitionedCallҐ-positional_encoder_16/StatefulPartitionedCallҐ$sequential_6/StatefulPartitionedCallҐ,tubelet_embedding_32/StatefulPartitionedCallҐ,tubelet_embedding_33/StatefulPartitionedCallМ
/tf.__operators__.getitem_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               в
)tf.__operators__.getitem_37/strided_sliceStridedSliceinput_178tf.__operators__.getitem_37/strided_slice/stack:output:0:tf.__operators__.getitem_37/strided_slice/stack_1:output:0:tf.__operators__.getitem_37/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_maskМ
/tf.__operators__.getitem_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               в
)tf.__operators__.getitem_36/strided_sliceStridedSliceinput_178tf.__operators__.getitem_36/strided_slice/stack:output:0:tf.__operators__.getitem_36/strided_slice/stack_1:output:0:tf.__operators__.getitem_36/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_mask“
,tubelet_embedding_32/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_36/strided_slice:output:0tubelet_embedding_32_495373tubelet_embedding_32_495375tubelet_embedding_32_495377tubelet_embedding_32_495379tubelet_embedding_32_495381tubelet_embedding_32_495383*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_tubelet_embedding_32_layer_call_and_return_conditional_losses_494440“
,tubelet_embedding_33/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_37/strided_slice:output:0tubelet_embedding_33_495386tubelet_embedding_33_495388tubelet_embedding_33_495390tubelet_embedding_33_495392tubelet_embedding_33_495394tubelet_embedding_33_495396*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_tubelet_embedding_33_layer_call_and_return_conditional_losses_494490µ
concatenate_13/PartitionedCallPartitionedCall5tubelet_embedding_32/StatefulPartitionedCall:output:05tubelet_embedding_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *S
fNRL
J__inference_concatenate_13_layer_call_and_return_conditional_losses_494511ќ
-positional_encoder_16/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0positional_encoder_16_495400positional_encoder_16_495402*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Z
fURS
Q__inference_positional_encoder_16_layer_call_and_return_conditional_losses_494525в
.layer_normalization_18/StatefulPartitionedCallStatefulPartitionedCall6positional_encoder_16/StatefulPartitionedCall:output:0layer_normalization_18_495405layer_normalization_18_495407*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_18_layer_call_and_return_conditional_losses_494553€
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_18/StatefulPartitionedCall:output:07layer_normalization_18/StatefulPartitionedCall:output:0multi_head_attention_6_495410multi_head_attention_6_495412multi_head_attention_6_495414multi_head_attention_6_495416multi_head_attention_6_495418multi_head_attention_6_495420multi_head_attention_6_495422multi_head_attention_6_495424*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:€€€€€€€€€(А:€€€€€€€€€((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_494891®
add_12/PartitionedCallPartitionedCall7multi_head_attention_6/StatefulPartitionedCall:output:06positional_encoder_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_494620Ћ
.layer_normalization_19/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0layer_normalization_19_495429layer_normalization_19_495431*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_19_layer_call_and_return_conditional_losses_494644ї
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_19/StatefulPartitionedCall:output:0sequential_6_495434sequential_6_495436*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494342З
add_13/PartitionedCallPartitionedCall-sequential_6/StatefulPartitionedCall:output:0add_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *K
fFRD
B__inference_add_13_layer_call_and_return_conditional_losses_494661Ћ
.layer_normalization_20/StatefulPartitionedCallStatefulPartitionedCalladd_13/PartitionedCall:output:0layer_normalization_20_495440layer_normalization_20_495442*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_20_layer_call_and_return_conditional_losses_494685У
*global_average_pooling1d_6/PartitionedCallPartitionedCall7layer_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *_
fZRX
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_494386Ґ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_6/PartitionedCall:output:0dense_13_495446dense_13_495448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_494702x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€в
NoOpNoOp!^dense_13/StatefulPartitionedCall/^layer_normalization_18/StatefulPartitionedCall/^layer_normalization_19/StatefulPartitionedCall/^layer_normalization_20/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall.^positional_encoder_16/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall-^tubelet_embedding_32/StatefulPartitionedCall-^tubelet_embedding_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2`
.layer_normalization_18/StatefulPartitionedCall.layer_normalization_18/StatefulPartitionedCall2`
.layer_normalization_19/StatefulPartitionedCall.layer_normalization_19/StatefulPartitionedCall2`
.layer_normalization_20/StatefulPartitionedCall.layer_normalization_20/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2^
-positional_encoder_16/StatefulPartitionedCall-positional_encoder_16/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2\
,tubelet_embedding_32/StatefulPartitionedCall,tubelet_embedding_32/StatefulPartitionedCall2\
,tubelet_embedding_33/StatefulPartitionedCall,tubelet_embedding_33/StatefulPartitionedCall:] Y
3
_output_shapes!
:€€€€€€€€€

"
_user_specified_name
input_17: 

_output_shapes
:(
н+
Ч
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_494595	
query	
valueC
+query_einsum_einsum_readvariableop_resource:АА4
!query_add_readvariableop_resource:	АA
)key_einsum_einsum_readvariableop_resource:АА2
key_add_readvariableop_resource:	АC
+value_einsum_einsum_readvariableop_resource:АА4
!value_add_readvariableop_resource:	АN
6attention_output_einsum_einsum_readvariableop_resource:АА;
,attention_output_add_readvariableop_resource:	А
identity

identity_1ИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpФ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde{
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АР
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abdew
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes
:	А*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АФ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationabc,cde->abde{
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€(АJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€(АП
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€((*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€((q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€((¶
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€(А*
equationacbe,aecd->abcd™
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А*
equationabcd,cde->abeН
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аr

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€((Ў
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€(А:€€€€€€€€€(А: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namevalue
Д
W
;__inference_global_average_pooling1d_6_layer_call_fn_496585

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *_
fZRX
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_494386i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Џ
h
L__inference_max_pooling3d_64_layer_call_and_return_conditional_losses_494190

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ё
l
P__inference_average_pooling3d_32_layer_call_and_return_conditional_losses_494214

inputs
identityЊ
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
у
M
1__inference_max_pooling3d_67_layer_call_fn_496655

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *U
fPRN
L__inference_max_pooling3d_67_layer_call_and_return_conditional_losses_494238Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
О
r
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_494386

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ю
•
-__inference_sequential_6_layer_call_fn_494312
dense_12_input
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCalldense_12_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494305t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:€€€€€€€€€(А
(
_user_specified_namedense_12_input
’
l
B__inference_add_13_layer_call_and_return_conditional_losses_494661

inputs
inputs_1
identityU
addAddV2inputsinputs_1*
T0*,
_output_shapes
:€€€€€€€€€(АT
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€(А:€€€€€€€€€(А:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
ж
Э
-__inference_sequential_6_layer_call_fn_496461

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494342t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
з
„
H__inference_sequential_6_layer_call_and_return_conditional_losses_494376
dense_12_input#
dense_12_494370:
АА
dense_12_494372:	А
identityИҐ dense_12/StatefulPartitionedCallВ
 dense_12/StatefulPartitionedCallStatefulPartitionedCalldense_12_inputdense_12_494370dense_12_494372*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_494298}
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аi
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:\ X
,
_output_shapes
:€€€€€€€€€(А
(
_user_specified_namedense_12_input
ѕ	
і
5__inference_tubelet_embedding_33_layer_call_fn_496172

videos%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@А
	unknown_4:	А
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallvideosunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_tubelet_embedding_33_layer_call_and_return_conditional_losses_494490t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_namevideos
в
И
7__inference_multi_head_attention_6_layer_call_fn_496321	
query	
value
unknown:АА
	unknown_0:	А!
	unknown_1:АА
	unknown_2:	А!
	unknown_3:АА
	unknown_4:	А!
	unknown_5:АА
	unknown_6:	А
identity

identity_1ИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:€€€€€€€€€(А:€€€€€€€€€((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_494891t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аy

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:€€€€€€€€€((`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€(А:€€€€€€€€€(А: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€(А

_user_specified_namevalue
Џ
h
L__inference_max_pooling3d_67_layer_call_and_return_conditional_losses_496660

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ћ	
ц
D__inference_dense_13_layer_call_and_return_conditional_losses_494702

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѕ
ѕ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494305

inputs#
dense_12_494299:
АА
dense_12_494301:	А
identityИҐ dense_12/StatefulPartitionedCallъ
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_494299dense_12_494301*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_494298}
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аi
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
ю
У
R__inference_layer_normalization_18_layer_call_and_return_conditional_losses_496273

inputs4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(М
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ж
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0В
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АА
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
Б
љ
(__inference_model_6_layer_call_fn_495596

inputs%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@А
	unknown_4:	А'
	unknown_5: 
	unknown_6: '
	unknown_7: @
	unknown_8:@(
	unknown_9:@А

unknown_10:	А

unknown_11

unknown_12:	(А

unknown_13:	А

unknown_14:	А"

unknown_15:АА

unknown_16:	А"

unknown_17:АА

unknown_18:	А"

unknown_19:АА

unknown_20:	А"

unknown_21:АА

unknown_22:	А

unknown_23:	А

unknown_24:	А

unknown_25:
АА

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8В *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_495136o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_nameinputs: 

_output_shapes
:(
Џ
h
L__inference_max_pooling3d_64_layer_call_and_return_conditional_losses_496620

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ж
Э
-__inference_sequential_6_layer_call_fn_496452

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494305t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
ѕ
ѕ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494342

inputs#
dense_12_494336:
АА
dense_12_494338:	А
identityИҐ dense_12/StatefulPartitionedCallъ
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_494336dense_12_494338*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_494298}
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аi
NoOpNoOp!^dense_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
у
M
1__inference_max_pooling3d_65_layer_call_fn_496625

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *U
fPRN
L__inference_max_pooling3d_65_layer_call_and_return_conditional_losses_494202Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
®‘
∞6
__inference__traced_save_497047
file_prefix;
7savev2_layer_normalization_18_gamma_read_readvariableop:
6savev2_layer_normalization_18_beta_read_readvariableop;
7savev2_layer_normalization_19_gamma_read_readvariableop:
6savev2_layer_normalization_19_beta_read_readvariableop;
7savev2_layer_normalization_20_gamma_read_readvariableop:
6savev2_layer_normalization_20_beta_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_tubelet_embedding_32_conv3d_96_kernel_read_readvariableopB
>savev2_tubelet_embedding_32_conv3d_96_bias_read_readvariableopD
@savev2_tubelet_embedding_32_conv3d_97_kernel_read_readvariableopB
>savev2_tubelet_embedding_32_conv3d_97_bias_read_readvariableopD
@savev2_tubelet_embedding_32_conv3d_98_kernel_read_readvariableopB
>savev2_tubelet_embedding_32_conv3d_98_bias_read_readvariableopD
@savev2_tubelet_embedding_33_conv3d_99_kernel_read_readvariableopB
>savev2_tubelet_embedding_33_conv3d_99_bias_read_readvariableopE
Asavev2_tubelet_embedding_33_conv3d_100_kernel_read_readvariableopC
?savev2_tubelet_embedding_33_conv3d_100_bias_read_readvariableopE
Asavev2_tubelet_embedding_33_conv3d_101_kernel_read_readvariableopC
?savev2_tubelet_embedding_33_conv3d_101_bias_read_readvariableopI
Esavev2_positional_encoder_16_embedding_embeddings_read_readvariableopB
>savev2_multi_head_attention_6_query_kernel_read_readvariableop@
<savev2_multi_head_attention_6_query_bias_read_readvariableop@
<savev2_multi_head_attention_6_key_kernel_read_readvariableop>
:savev2_multi_head_attention_6_key_bias_read_readvariableopB
>savev2_multi_head_attention_6_value_kernel_read_readvariableop@
<savev2_multi_head_attention_6_value_bias_read_readvariableopM
Isavev2_multi_head_attention_6_attention_output_kernel_read_readvariableopK
Gsavev2_multi_head_attention_6_attention_output_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopB
>savev2_adam_layer_normalization_18_gamma_m_read_readvariableopA
=savev2_adam_layer_normalization_18_beta_m_read_readvariableopB
>savev2_adam_layer_normalization_19_gamma_m_read_readvariableopA
=savev2_adam_layer_normalization_19_beta_m_read_readvariableopB
>savev2_adam_layer_normalization_20_gamma_m_read_readvariableopA
=savev2_adam_layer_normalization_20_beta_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableopK
Gsavev2_adam_tubelet_embedding_32_conv3d_96_kernel_m_read_readvariableopI
Esavev2_adam_tubelet_embedding_32_conv3d_96_bias_m_read_readvariableopK
Gsavev2_adam_tubelet_embedding_32_conv3d_97_kernel_m_read_readvariableopI
Esavev2_adam_tubelet_embedding_32_conv3d_97_bias_m_read_readvariableopK
Gsavev2_adam_tubelet_embedding_32_conv3d_98_kernel_m_read_readvariableopI
Esavev2_adam_tubelet_embedding_32_conv3d_98_bias_m_read_readvariableopK
Gsavev2_adam_tubelet_embedding_33_conv3d_99_kernel_m_read_readvariableopI
Esavev2_adam_tubelet_embedding_33_conv3d_99_bias_m_read_readvariableopL
Hsavev2_adam_tubelet_embedding_33_conv3d_100_kernel_m_read_readvariableopJ
Fsavev2_adam_tubelet_embedding_33_conv3d_100_bias_m_read_readvariableopL
Hsavev2_adam_tubelet_embedding_33_conv3d_101_kernel_m_read_readvariableopJ
Fsavev2_adam_tubelet_embedding_33_conv3d_101_bias_m_read_readvariableopP
Lsavev2_adam_positional_encoder_16_embedding_embeddings_m_read_readvariableopI
Esavev2_adam_multi_head_attention_6_query_kernel_m_read_readvariableopG
Csavev2_adam_multi_head_attention_6_query_bias_m_read_readvariableopG
Csavev2_adam_multi_head_attention_6_key_kernel_m_read_readvariableopE
Asavev2_adam_multi_head_attention_6_key_bias_m_read_readvariableopI
Esavev2_adam_multi_head_attention_6_value_kernel_m_read_readvariableopG
Csavev2_adam_multi_head_attention_6_value_bias_m_read_readvariableopT
Psavev2_adam_multi_head_attention_6_attention_output_kernel_m_read_readvariableopR
Nsavev2_adam_multi_head_attention_6_attention_output_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableopB
>savev2_adam_layer_normalization_18_gamma_v_read_readvariableopA
=savev2_adam_layer_normalization_18_beta_v_read_readvariableopB
>savev2_adam_layer_normalization_19_gamma_v_read_readvariableopA
=savev2_adam_layer_normalization_19_beta_v_read_readvariableopB
>savev2_adam_layer_normalization_20_gamma_v_read_readvariableopA
=savev2_adam_layer_normalization_20_beta_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableopK
Gsavev2_adam_tubelet_embedding_32_conv3d_96_kernel_v_read_readvariableopI
Esavev2_adam_tubelet_embedding_32_conv3d_96_bias_v_read_readvariableopK
Gsavev2_adam_tubelet_embedding_32_conv3d_97_kernel_v_read_readvariableopI
Esavev2_adam_tubelet_embedding_32_conv3d_97_bias_v_read_readvariableopK
Gsavev2_adam_tubelet_embedding_32_conv3d_98_kernel_v_read_readvariableopI
Esavev2_adam_tubelet_embedding_32_conv3d_98_bias_v_read_readvariableopK
Gsavev2_adam_tubelet_embedding_33_conv3d_99_kernel_v_read_readvariableopI
Esavev2_adam_tubelet_embedding_33_conv3d_99_bias_v_read_readvariableopL
Hsavev2_adam_tubelet_embedding_33_conv3d_100_kernel_v_read_readvariableopJ
Fsavev2_adam_tubelet_embedding_33_conv3d_100_bias_v_read_readvariableopL
Hsavev2_adam_tubelet_embedding_33_conv3d_101_kernel_v_read_readvariableopJ
Fsavev2_adam_tubelet_embedding_33_conv3d_101_bias_v_read_readvariableopP
Lsavev2_adam_positional_encoder_16_embedding_embeddings_v_read_readvariableopI
Esavev2_adam_multi_head_attention_6_query_kernel_v_read_readvariableopG
Csavev2_adam_multi_head_attention_6_query_bias_v_read_readvariableopG
Csavev2_adam_multi_head_attention_6_key_kernel_v_read_readvariableopE
Asavev2_adam_multi_head_attention_6_key_bias_v_read_readvariableopI
Esavev2_adam_multi_head_attention_6_value_kernel_v_read_readvariableopG
Csavev2_adam_multi_head_attention_6_value_bias_v_read_readvariableopT
Psavev2_adam_multi_head_attention_6_attention_output_kernel_v_read_readvariableopR
Nsavev2_adam_multi_head_attention_6_attention_output_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop
savev2_const_1

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Э2
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:g*
dtype0*∆1
valueЉ1Bє1gB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЊ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:g*
dtype0*г
valueўB÷gB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¬4
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_layer_normalization_18_gamma_read_readvariableop6savev2_layer_normalization_18_beta_read_readvariableop7savev2_layer_normalization_19_gamma_read_readvariableop6savev2_layer_normalization_19_beta_read_readvariableop7savev2_layer_normalization_20_gamma_read_readvariableop6savev2_layer_normalization_20_beta_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_tubelet_embedding_32_conv3d_96_kernel_read_readvariableop>savev2_tubelet_embedding_32_conv3d_96_bias_read_readvariableop@savev2_tubelet_embedding_32_conv3d_97_kernel_read_readvariableop>savev2_tubelet_embedding_32_conv3d_97_bias_read_readvariableop@savev2_tubelet_embedding_32_conv3d_98_kernel_read_readvariableop>savev2_tubelet_embedding_32_conv3d_98_bias_read_readvariableop@savev2_tubelet_embedding_33_conv3d_99_kernel_read_readvariableop>savev2_tubelet_embedding_33_conv3d_99_bias_read_readvariableopAsavev2_tubelet_embedding_33_conv3d_100_kernel_read_readvariableop?savev2_tubelet_embedding_33_conv3d_100_bias_read_readvariableopAsavev2_tubelet_embedding_33_conv3d_101_kernel_read_readvariableop?savev2_tubelet_embedding_33_conv3d_101_bias_read_readvariableopEsavev2_positional_encoder_16_embedding_embeddings_read_readvariableop>savev2_multi_head_attention_6_query_kernel_read_readvariableop<savev2_multi_head_attention_6_query_bias_read_readvariableop<savev2_multi_head_attention_6_key_kernel_read_readvariableop:savev2_multi_head_attention_6_key_bias_read_readvariableop>savev2_multi_head_attention_6_value_kernel_read_readvariableop<savev2_multi_head_attention_6_value_bias_read_readvariableopIsavev2_multi_head_attention_6_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_6_attention_output_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adam_layer_normalization_18_gamma_m_read_readvariableop=savev2_adam_layer_normalization_18_beta_m_read_readvariableop>savev2_adam_layer_normalization_19_gamma_m_read_readvariableop=savev2_adam_layer_normalization_19_beta_m_read_readvariableop>savev2_adam_layer_normalization_20_gamma_m_read_readvariableop=savev2_adam_layer_normalization_20_beta_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableopGsavev2_adam_tubelet_embedding_32_conv3d_96_kernel_m_read_readvariableopEsavev2_adam_tubelet_embedding_32_conv3d_96_bias_m_read_readvariableopGsavev2_adam_tubelet_embedding_32_conv3d_97_kernel_m_read_readvariableopEsavev2_adam_tubelet_embedding_32_conv3d_97_bias_m_read_readvariableopGsavev2_adam_tubelet_embedding_32_conv3d_98_kernel_m_read_readvariableopEsavev2_adam_tubelet_embedding_32_conv3d_98_bias_m_read_readvariableopGsavev2_adam_tubelet_embedding_33_conv3d_99_kernel_m_read_readvariableopEsavev2_adam_tubelet_embedding_33_conv3d_99_bias_m_read_readvariableopHsavev2_adam_tubelet_embedding_33_conv3d_100_kernel_m_read_readvariableopFsavev2_adam_tubelet_embedding_33_conv3d_100_bias_m_read_readvariableopHsavev2_adam_tubelet_embedding_33_conv3d_101_kernel_m_read_readvariableopFsavev2_adam_tubelet_embedding_33_conv3d_101_bias_m_read_readvariableopLsavev2_adam_positional_encoder_16_embedding_embeddings_m_read_readvariableopEsavev2_adam_multi_head_attention_6_query_kernel_m_read_readvariableopCsavev2_adam_multi_head_attention_6_query_bias_m_read_readvariableopCsavev2_adam_multi_head_attention_6_key_kernel_m_read_readvariableopAsavev2_adam_multi_head_attention_6_key_bias_m_read_readvariableopEsavev2_adam_multi_head_attention_6_value_kernel_m_read_readvariableopCsavev2_adam_multi_head_attention_6_value_bias_m_read_readvariableopPsavev2_adam_multi_head_attention_6_attention_output_kernel_m_read_readvariableopNsavev2_adam_multi_head_attention_6_attention_output_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop>savev2_adam_layer_normalization_18_gamma_v_read_readvariableop=savev2_adam_layer_normalization_18_beta_v_read_readvariableop>savev2_adam_layer_normalization_19_gamma_v_read_readvariableop=savev2_adam_layer_normalization_19_beta_v_read_readvariableop>savev2_adam_layer_normalization_20_gamma_v_read_readvariableop=savev2_adam_layer_normalization_20_beta_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableopGsavev2_adam_tubelet_embedding_32_conv3d_96_kernel_v_read_readvariableopEsavev2_adam_tubelet_embedding_32_conv3d_96_bias_v_read_readvariableopGsavev2_adam_tubelet_embedding_32_conv3d_97_kernel_v_read_readvariableopEsavev2_adam_tubelet_embedding_32_conv3d_97_bias_v_read_readvariableopGsavev2_adam_tubelet_embedding_32_conv3d_98_kernel_v_read_readvariableopEsavev2_adam_tubelet_embedding_32_conv3d_98_bias_v_read_readvariableopGsavev2_adam_tubelet_embedding_33_conv3d_99_kernel_v_read_readvariableopEsavev2_adam_tubelet_embedding_33_conv3d_99_bias_v_read_readvariableopHsavev2_adam_tubelet_embedding_33_conv3d_100_kernel_v_read_readvariableopFsavev2_adam_tubelet_embedding_33_conv3d_100_bias_v_read_readvariableopHsavev2_adam_tubelet_embedding_33_conv3d_101_kernel_v_read_readvariableopFsavev2_adam_tubelet_embedding_33_conv3d_101_bias_v_read_readvariableopLsavev2_adam_positional_encoder_16_embedding_embeddings_v_read_readvariableopEsavev2_adam_multi_head_attention_6_query_kernel_v_read_readvariableopCsavev2_adam_multi_head_attention_6_query_bias_v_read_readvariableopCsavev2_adam_multi_head_attention_6_key_kernel_v_read_readvariableopAsavev2_adam_multi_head_attention_6_key_bias_v_read_readvariableopEsavev2_adam_multi_head_attention_6_value_kernel_v_read_readvariableopCsavev2_adam_multi_head_attention_6_value_bias_v_read_readvariableopPsavev2_adam_multi_head_attention_6_attention_output_kernel_v_read_readvariableopNsavev2_adam_multi_head_attention_6_attention_output_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *u
dtypesk
i2g	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*т
_input_shapesа
Ё: :А:А:А:А:А:А:	А:: : : : : : : : @:@:@А:А: : : @:@:@А:А:	(А:АА:	А:АА:	А:АА:	А:АА:А:
АА:А: : : : :А:А:А:А:А:А:	А:: : : @:@:@А:А: : : @:@:@А:А:	(А:АА:	А:АА:	А:АА:	А:АА:А:
АА:А:А:А:А:А:А:А:	А:: : : @:@:@А:А: : : @:@:@А:А:	(А:АА:	А:АА:	А:АА:	А:АА:А:
АА:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
: @: 

_output_shapes
:@:1-
+
_output_shapes
:@А:!

_output_shapes	
:А:0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
: @: 

_output_shapes
:@:1-
+
_output_shapes
:@А:!

_output_shapes	
:А:%!

_output_shapes
:	(А:*&
$
_output_shapes
:АА:%!

_output_shapes
:	А:*&
$
_output_shapes
:АА:%!

_output_shapes
:	А:*&
$
_output_shapes
:АА:% !

_output_shapes
:	А:*!&
$
_output_shapes
:АА:!"

_output_shapes	
:А:&#"
 
_output_shapes
:
АА:!$

_output_shapes	
:А:%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :!)

_output_shapes	
:А:!*

_output_shapes	
:А:!+

_output_shapes	
:А:!,

_output_shapes	
:А:!-

_output_shapes	
:А:!.

_output_shapes	
:А:%/!

_output_shapes
:	А: 0

_output_shapes
::01,
*
_output_shapes
: : 2

_output_shapes
: :03,
*
_output_shapes
: @: 4

_output_shapes
:@:15-
+
_output_shapes
:@А:!6

_output_shapes	
:А:07,
*
_output_shapes
: : 8

_output_shapes
: :09,
*
_output_shapes
: @: :

_output_shapes
:@:1;-
+
_output_shapes
:@А:!<

_output_shapes	
:А:%=!

_output_shapes
:	(А:*>&
$
_output_shapes
:АА:%?!

_output_shapes
:	А:*@&
$
_output_shapes
:АА:%A!

_output_shapes
:	А:*B&
$
_output_shapes
:АА:%C!

_output_shapes
:	А:*D&
$
_output_shapes
:АА:!E

_output_shapes	
:А:&F"
 
_output_shapes
:
АА:!G

_output_shapes	
:А:!H

_output_shapes	
:А:!I

_output_shapes	
:А:!J

_output_shapes	
:А:!K

_output_shapes	
:А:!L

_output_shapes	
:А:!M

_output_shapes	
:А:%N!

_output_shapes
:	А: O

_output_shapes
::0P,
*
_output_shapes
: : Q

_output_shapes
: :0R,
*
_output_shapes
: @: S

_output_shapes
:@:1T-
+
_output_shapes
:@А:!U

_output_shapes	
:А:0V,
*
_output_shapes
: : W

_output_shapes
: :0X,
*
_output_shapes
: @: Y

_output_shapes
:@:1Z-
+
_output_shapes
:@А:![

_output_shapes	
:А:%\!

_output_shapes
:	(А:*]&
$
_output_shapes
:АА:%^!

_output_shapes
:	А:*_&
$
_output_shapes
:АА:%`!

_output_shapes
:	А:*a&
$
_output_shapes
:АА:%b!

_output_shapes
:	А:*c&
$
_output_shapes
:АА:!d

_output_shapes	
:А:&e"
 
_output_shapes
:
АА:!f

_output_shapes	
:А:g

_output_shapes
: 
Ё
n
B__inference_add_12_layer_call_and_return_conditional_losses_496412
inputs_0
inputs_1
identityW
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:€€€€€€€€€(АT
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€(А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€(А:€€€€€€€€€(А:V R
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€(А
"
_user_specified_name
inputs/1
я"
ю
D__inference_dense_12_layer_call_and_return_conditional_losses_496717

inputs5
!tensordot_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АК
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Л
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:АY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Д
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(АO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АP
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?v
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АX
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АO

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€(Аd

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аb
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
ю
У
R__inference_layer_normalization_20_layer_call_and_return_conditional_losses_496580

inputs4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(М
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ж
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0В
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АА
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
г[
э
C__inference_model_6_layer_call_and_return_conditional_losses_494709

inputs9
tubelet_embedding_32_494441: )
tubelet_embedding_32_494443: 9
tubelet_embedding_32_494445: @)
tubelet_embedding_32_494447:@:
tubelet_embedding_32_494449:@А*
tubelet_embedding_32_494451:	А9
tubelet_embedding_33_494491: )
tubelet_embedding_33_494493: 9
tubelet_embedding_33_494495: @)
tubelet_embedding_33_494497:@:
tubelet_embedding_33_494499:@А*
tubelet_embedding_33_494501:	А 
positional_encoder_16_494526/
positional_encoder_16_494528:	(А,
layer_normalization_18_494554:	А,
layer_normalization_18_494556:	А5
multi_head_attention_6_494596:АА0
multi_head_attention_6_494598:	А5
multi_head_attention_6_494600:АА0
multi_head_attention_6_494602:	А5
multi_head_attention_6_494604:АА0
multi_head_attention_6_494606:	А5
multi_head_attention_6_494608:АА,
multi_head_attention_6_494610:	А,
layer_normalization_19_494645:	А,
layer_normalization_19_494647:	А'
sequential_6_494650:
АА"
sequential_6_494652:	А,
layer_normalization_20_494686:	А,
layer_normalization_20_494688:	А"
dense_13_494703:	А
dense_13_494705:
identityИҐ dense_13/StatefulPartitionedCallҐ.layer_normalization_18/StatefulPartitionedCallҐ.layer_normalization_19/StatefulPartitionedCallҐ.layer_normalization_20/StatefulPartitionedCallҐ.multi_head_attention_6/StatefulPartitionedCallҐ-positional_encoder_16/StatefulPartitionedCallҐ$sequential_6/StatefulPartitionedCallҐ,tubelet_embedding_32/StatefulPartitionedCallҐ,tubelet_embedding_33/StatefulPartitionedCallМ
/tf.__operators__.getitem_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               а
)tf.__operators__.getitem_37/strided_sliceStridedSliceinputs8tf.__operators__.getitem_37/strided_slice/stack:output:0:tf.__operators__.getitem_37/strided_slice/stack_1:output:0:tf.__operators__.getitem_37/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_maskМ
/tf.__operators__.getitem_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               а
)tf.__operators__.getitem_36/strided_sliceStridedSliceinputs8tf.__operators__.getitem_36/strided_slice/stack:output:0:tf.__operators__.getitem_36/strided_slice/stack_1:output:0:tf.__operators__.getitem_36/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_mask“
,tubelet_embedding_32/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_36/strided_slice:output:0tubelet_embedding_32_494441tubelet_embedding_32_494443tubelet_embedding_32_494445tubelet_embedding_32_494447tubelet_embedding_32_494449tubelet_embedding_32_494451*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_tubelet_embedding_32_layer_call_and_return_conditional_losses_494440“
,tubelet_embedding_33/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_37/strided_slice:output:0tubelet_embedding_33_494491tubelet_embedding_33_494493tubelet_embedding_33_494495tubelet_embedding_33_494497tubelet_embedding_33_494499tubelet_embedding_33_494501*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_tubelet_embedding_33_layer_call_and_return_conditional_losses_494490µ
concatenate_13/PartitionedCallPartitionedCall5tubelet_embedding_32/StatefulPartitionedCall:output:05tubelet_embedding_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *S
fNRL
J__inference_concatenate_13_layer_call_and_return_conditional_losses_494511ќ
-positional_encoder_16/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0positional_encoder_16_494526positional_encoder_16_494528*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Z
fURS
Q__inference_positional_encoder_16_layer_call_and_return_conditional_losses_494525в
.layer_normalization_18/StatefulPartitionedCallStatefulPartitionedCall6positional_encoder_16/StatefulPartitionedCall:output:0layer_normalization_18_494554layer_normalization_18_494556*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_18_layer_call_and_return_conditional_losses_494553€
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_18/StatefulPartitionedCall:output:07layer_normalization_18/StatefulPartitionedCall:output:0multi_head_attention_6_494596multi_head_attention_6_494598multi_head_attention_6_494600multi_head_attention_6_494602multi_head_attention_6_494604multi_head_attention_6_494606multi_head_attention_6_494608multi_head_attention_6_494610*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:€€€€€€€€€(А:€€€€€€€€€((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_494595®
add_12/PartitionedCallPartitionedCall7multi_head_attention_6/StatefulPartitionedCall:output:06positional_encoder_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_494620Ћ
.layer_normalization_19/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0layer_normalization_19_494645layer_normalization_19_494647*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_19_layer_call_and_return_conditional_losses_494644ї
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_19/StatefulPartitionedCall:output:0sequential_6_494650sequential_6_494652*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494305З
add_13/PartitionedCallPartitionedCall-sequential_6/StatefulPartitionedCall:output:0add_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *K
fFRD
B__inference_add_13_layer_call_and_return_conditional_losses_494661Ћ
.layer_normalization_20/StatefulPartitionedCallStatefulPartitionedCalladd_13/PartitionedCall:output:0layer_normalization_20_494686layer_normalization_20_494688*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_20_layer_call_and_return_conditional_losses_494685У
*global_average_pooling1d_6/PartitionedCallPartitionedCall7layer_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *_
fZRX
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_494386Ґ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_6/PartitionedCall:output:0dense_13_494703dense_13_494705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_494702x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€в
NoOpNoOp!^dense_13/StatefulPartitionedCall/^layer_normalization_18/StatefulPartitionedCall/^layer_normalization_19/StatefulPartitionedCall/^layer_normalization_20/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall.^positional_encoder_16/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall-^tubelet_embedding_32/StatefulPartitionedCall-^tubelet_embedding_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2`
.layer_normalization_18/StatefulPartitionedCall.layer_normalization_18/StatefulPartitionedCall2`
.layer_normalization_19/StatefulPartitionedCall.layer_normalization_19/StatefulPartitionedCall2`
.layer_normalization_20/StatefulPartitionedCall.layer_normalization_20/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2^
-positional_encoder_16/StatefulPartitionedCall-positional_encoder_16/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2\
,tubelet_embedding_32/StatefulPartitionedCall,tubelet_embedding_32/StatefulPartitionedCall2\
,tubelet_embedding_33/StatefulPartitionedCall,tubelet_embedding_33/StatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_nameinputs: 

_output_shapes
:(
О
r
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_496591

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞-
Ћ
P__inference_tubelet_embedding_33_layer_call_and_return_conditional_losses_494490

videosF
(conv3d_99_conv3d_readvariableop_resource: 7
)conv3d_99_biasadd_readvariableop_resource: G
)conv3d_100_conv3d_readvariableop_resource: @8
*conv3d_100_biasadd_readvariableop_resource:@H
)conv3d_101_conv3d_readvariableop_resource:@А9
*conv3d_101_biasadd_readvariableop_resource:	А
identityИҐ!conv3d_100/BiasAdd/ReadVariableOpҐ conv3d_100/Conv3D/ReadVariableOpҐ!conv3d_101/BiasAdd/ReadVariableOpҐ conv3d_101/Conv3D/ReadVariableOpҐ conv3d_99/BiasAdd/ReadVariableOpҐconv3d_99/Conv3D/ReadVariableOpФ
conv3d_99/Conv3D/ReadVariableOpReadVariableOp(conv3d_99_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0≤
conv3d_99/Conv3DConv3Dvideos'conv3d_99/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
paddingSAME*
strides	
Ж
 conv3d_99/BiasAdd/ReadVariableOpReadVariableOp)conv3d_99_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Я
conv3d_99/BiasAddBiasAddconv3d_99/Conv3D:output:0(conv3d_99/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
 p
conv3d_99/ReluReluconv3d_99/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
 Ѕ
max_pooling3d_66/MaxPool3D	MaxPool3Dconv3d_99/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
 *
ksize	
*
paddingVALID*
strides	
Ц
 conv3d_100/Conv3D/ReadVariableOpReadVariableOp)conv3d_100_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0—
conv3d_100/Conv3DConv3D#max_pooling3d_66/MaxPool3D:output:0(conv3d_100/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
paddingSAME*
strides	
И
!conv3d_100/BiasAdd/ReadVariableOpReadVariableOp*conv3d_100_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ґ
conv3d_100/BiasAddBiasAddconv3d_100/Conv3D:output:0)conv3d_100/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€
@r
conv3d_100/ReluReluconv3d_100/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€
@¬
max_pooling3d_67/MaxPool3D	MaxPool3Dconv3d_100/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€
@*
ksize	
*
paddingVALID*
strides	
Ч
 conv3d_101/Conv3D/ReadVariableOpReadVariableOp)conv3d_101_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0“
conv3d_101/Conv3DConv3D#max_pooling3d_67/MaxPool3D:output:0(conv3d_101/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
paddingSAME*
strides	
Й
!conv3d_101/BiasAdd/ReadVariableOpReadVariableOp*conv3d_101_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0£
conv3d_101/BiasAddBiasAddconv3d_101/Conv3D:output:0)conv3d_101/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А≈
average_pooling3d_33/AvgPool3D	AvgPool3Dconv3d_101/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€
А*
ksize	
*
paddingVALID*
strides	
g
reshape_33/ShapeShape'average_pooling3d_33/AvgPool3D:output:0*
T0*
_output_shapes
:h
reshape_33/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_33/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_33/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
reshape_33/strided_sliceStridedSlicereshape_33/Shape:output:0'reshape_33/strided_slice/stack:output:0)reshape_33/strided_slice/stack_1:output:0)reshape_33/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
reshape_33/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€]
reshape_33/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Аї
reshape_33/Reshape/shapePack!reshape_33/strided_slice:output:0#reshape_33/Reshape/shape/1:output:0#reshape_33/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:†
reshape_33/ReshapeReshape'average_pooling3d_33/AvgPool3D:output:0!reshape_33/Reshape/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аo
IdentityIdentityreshape_33/Reshape:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АЩ
NoOpNoOp"^conv3d_100/BiasAdd/ReadVariableOp!^conv3d_100/Conv3D/ReadVariableOp"^conv3d_101/BiasAdd/ReadVariableOp!^conv3d_101/Conv3D/ReadVariableOp!^conv3d_99/BiasAdd/ReadVariableOp ^conv3d_99/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€
: : : : : : 2F
!conv3d_100/BiasAdd/ReadVariableOp!conv3d_100/BiasAdd/ReadVariableOp2D
 conv3d_100/Conv3D/ReadVariableOp conv3d_100/Conv3D/ReadVariableOp2F
!conv3d_101/BiasAdd/ReadVariableOp!conv3d_101/BiasAdd/ReadVariableOp2D
 conv3d_101/Conv3D/ReadVariableOp conv3d_101/Conv3D/ReadVariableOp2D
 conv3d_99/BiasAdd/ReadVariableOp conv3d_99/BiasAdd/ReadVariableOp2B
conv3d_99/Conv3D/ReadVariableOpconv3d_99/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€

 
_user_specified_namevideos
ё
l
P__inference_average_pooling3d_32_layer_call_and_return_conditional_losses_496640

inputs
identityЊ
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я"
ю
D__inference_dense_12_layer_call_and_return_conditional_losses_494298

inputs5
!tensordot_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АК
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€Л
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:АY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Д
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(АO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АP
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?v
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€(АX
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€(АO

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€(Аd

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аb
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
н

Ў
Q__inference_positional_encoder_16_layer_call_and_return_conditional_losses_496242
encoded_tokens
unknown4
!embedding_embedding_lookup_496235:	(А
identityИҐembedding/embedding_lookupћ
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_496235unknown*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/496235*
_output_shapes
:	(А*
dtype0і
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/496235*
_output_shapes
:	(АЙ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	(АГ
addAddV2encoded_tokens.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(Аc
NoOpNoOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€(А:(: 28
embedding/embedding_lookupembedding/embedding_lookup:\ X
,
_output_shapes
:€€€€€€€€€(А
(
_user_specified_nameencoded_tokens: 

_output_shapes
:(
З
њ
(__inference_model_6_layer_call_fn_495272
input_17%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@А
	unknown_4:	А'
	unknown_5: 
	unknown_6: '
	unknown_7: @
	unknown_8:@(
	unknown_9:@А

unknown_10:	А

unknown_11

unknown_12:	(А

unknown_13:	А

unknown_14:	А"

unknown_15:АА

unknown_16:	А"

unknown_17:АА

unknown_18:	А"

unknown_19:АА

unknown_20:	А"

unknown_21:АА

unknown_22:	А

unknown_23:	А

unknown_24:	А

unknown_25:
АА

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8В *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_495136o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
3
_output_shapes!
:€€€€€€€€€

"
_user_specified_name
input_17: 

_output_shapes
:(
ю
У
R__inference_layer_normalization_19_layer_call_and_return_conditional_losses_496443

inputs4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:М
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€(М
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€(Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж5Б
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0Ж
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€(Аh
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0В
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аw
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€(Аg
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(АА
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
ў
t
J__inference_concatenate_13_layer_call_and_return_conditional_losses_494511

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :z
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:€€€€€€€€€(А\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:€€€€€€€€€(А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€(А:€€€€€€€€€(А:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
х
Ґ
7__inference_layer_normalization_19_layer_call_fn_496421

inputs
unknown:	А
	unknown_0:	А
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_19_layer_call_and_return_conditional_losses_494644t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€(А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€(А: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€(А
 
_user_specified_nameinputs
л[
€
C__inference_model_6_layer_call_and_return_conditional_losses_495362
input_179
tubelet_embedding_32_495283: )
tubelet_embedding_32_495285: 9
tubelet_embedding_32_495287: @)
tubelet_embedding_32_495289:@:
tubelet_embedding_32_495291:@А*
tubelet_embedding_32_495293:	А9
tubelet_embedding_33_495296: )
tubelet_embedding_33_495298: 9
tubelet_embedding_33_495300: @)
tubelet_embedding_33_495302:@:
tubelet_embedding_33_495304:@А*
tubelet_embedding_33_495306:	А 
positional_encoder_16_495310/
positional_encoder_16_495312:	(А,
layer_normalization_18_495315:	А,
layer_normalization_18_495317:	А5
multi_head_attention_6_495320:АА0
multi_head_attention_6_495322:	А5
multi_head_attention_6_495324:АА0
multi_head_attention_6_495326:	А5
multi_head_attention_6_495328:АА0
multi_head_attention_6_495330:	А5
multi_head_attention_6_495332:АА,
multi_head_attention_6_495334:	А,
layer_normalization_19_495339:	А,
layer_normalization_19_495341:	А'
sequential_6_495344:
АА"
sequential_6_495346:	А,
layer_normalization_20_495350:	А,
layer_normalization_20_495352:	А"
dense_13_495356:	А
dense_13_495358:
identityИҐ dense_13/StatefulPartitionedCallҐ.layer_normalization_18/StatefulPartitionedCallҐ.layer_normalization_19/StatefulPartitionedCallҐ.layer_normalization_20/StatefulPartitionedCallҐ.multi_head_attention_6/StatefulPartitionedCallҐ-positional_encoder_16/StatefulPartitionedCallҐ$sequential_6/StatefulPartitionedCallҐ,tubelet_embedding_32/StatefulPartitionedCallҐ,tubelet_embedding_33/StatefulPartitionedCallМ
/tf.__operators__.getitem_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               в
)tf.__operators__.getitem_37/strided_sliceStridedSliceinput_178tf.__operators__.getitem_37/strided_slice/stack:output:0:tf.__operators__.getitem_37/strided_slice/stack_1:output:0:tf.__operators__.getitem_37/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_maskМ
/tf.__operators__.getitem_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    О
1tf.__operators__.getitem_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   О
1tf.__operators__.getitem_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               в
)tf.__operators__.getitem_36/strided_sliceStridedSliceinput_178tf.__operators__.getitem_36/strided_slice/stack:output:0:tf.__operators__.getitem_36/strided_slice/stack_1:output:0:tf.__operators__.getitem_36/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:€€€€€€€€€
*

begin_mask*
end_mask“
,tubelet_embedding_32/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_36/strided_slice:output:0tubelet_embedding_32_495283tubelet_embedding_32_495285tubelet_embedding_32_495287tubelet_embedding_32_495289tubelet_embedding_32_495291tubelet_embedding_32_495293*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_tubelet_embedding_32_layer_call_and_return_conditional_losses_494440“
,tubelet_embedding_33/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_37/strided_slice:output:0tubelet_embedding_33_495296tubelet_embedding_33_495298tubelet_embedding_33_495300tubelet_embedding_33_495302tubelet_embedding_33_495304tubelet_embedding_33_495306*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8В *Y
fTRR
P__inference_tubelet_embedding_33_layer_call_and_return_conditional_losses_494490µ
concatenate_13/PartitionedCallPartitionedCall5tubelet_embedding_32/StatefulPartitionedCall:output:05tubelet_embedding_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *S
fNRL
J__inference_concatenate_13_layer_call_and_return_conditional_losses_494511ќ
-positional_encoder_16/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0positional_encoder_16_495310positional_encoder_16_495312*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Z
fURS
Q__inference_positional_encoder_16_layer_call_and_return_conditional_losses_494525в
.layer_normalization_18/StatefulPartitionedCallStatefulPartitionedCall6positional_encoder_16/StatefulPartitionedCall:output:0layer_normalization_18_495315layer_normalization_18_495317*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_18_layer_call_and_return_conditional_losses_494553€
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_18/StatefulPartitionedCall:output:07layer_normalization_18/StatefulPartitionedCall:output:0multi_head_attention_6_495320multi_head_attention_6_495322multi_head_attention_6_495324multi_head_attention_6_495326multi_head_attention_6_495328multi_head_attention_6_495330multi_head_attention_6_495332multi_head_attention_6_495334*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:€€€€€€€€€(А:€€€€€€€€€((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_494595®
add_12/PartitionedCallPartitionedCall7multi_head_attention_6/StatefulPartitionedCall:output:06positional_encoder_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_494620Ћ
.layer_normalization_19/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0layer_normalization_19_495339layer_normalization_19_495341*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_19_layer_call_and_return_conditional_losses_494644ї
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_19/StatefulPartitionedCall:output:0sequential_6_495344sequential_6_495346*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494305З
add_13/PartitionedCallPartitionedCall-sequential_6/StatefulPartitionedCall:output:0add_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *K
fFRD
B__inference_add_13_layer_call_and_return_conditional_losses_494661Ћ
.layer_normalization_20/StatefulPartitionedCallStatefulPartitionedCalladd_13/PartitionedCall:output:0layer_normalization_20_495350layer_normalization_20_495352*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€(А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *[
fVRT
R__inference_layer_normalization_20_layer_call_and_return_conditional_losses_494685У
*global_average_pooling1d_6/PartitionedCallPartitionedCall7layer_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8В *_
fZRX
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_494386Ґ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_6/PartitionedCall:output:0dense_13_495356dense_13_495358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_494702x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€в
NoOpNoOp!^dense_13/StatefulPartitionedCall/^layer_normalization_18/StatefulPartitionedCall/^layer_normalization_19/StatefulPartitionedCall/^layer_normalization_20/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall.^positional_encoder_16/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall-^tubelet_embedding_32/StatefulPartitionedCall-^tubelet_embedding_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2`
.layer_normalization_18/StatefulPartitionedCall.layer_normalization_18/StatefulPartitionedCall2`
.layer_normalization_19/StatefulPartitionedCall.layer_normalization_19/StatefulPartitionedCall2`
.layer_normalization_20/StatefulPartitionedCall.layer_normalization_20/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2^
-positional_encoder_16/StatefulPartitionedCall-positional_encoder_16/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2\
,tubelet_embedding_32/StatefulPartitionedCall,tubelet_embedding_32/StatefulPartitionedCall2\
,tubelet_embedding_33/StatefulPartitionedCall,tubelet_embedding_33/StatefulPartitionedCall:] Y
3
_output_shapes!
:€€€€€€€€€

"
_user_specified_name
input_17: 

_output_shapes
:("џL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*є
serving_default•
I
input_17=
serving_default_input_17:0€€€€€€€€€
<
dense_130
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:МИ
£
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
Д

projection
pool
projection2
	pool2
 projection3
	!pool4
"flatten
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
)
projection
*pool
+projection2
	,pool2
-projection3
	.pool4
/flatten
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
•
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
љ
<position_embedding
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
ƒ
Caxis
	Dgamma
Ebeta
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
О
L_query_dense
M
_key_dense
N_value_dense
O_softmax
P_dropout_layer
Q_output_dense
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
•
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
ƒ
^axis
	_gamma
`beta
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
—
glayer_with_weights-0
glayer-0
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_sequential
•
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
ƒ
taxis
	ugamma
vbeta
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
®
}	variables
~trainable_variables
regularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Гkernel
	Дbias
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	Лiter
Мbeta_1
Нbeta_2

Оdecay
Пlearning_rateDmфEmх_mц`mчumшvmщ	Гmъ	Дmы	Рmь	Сmэ	Тmю	Уm€	ФmА	ХmБ	ЦmВ	ЧmГ	ШmД	ЩmЕ	ЪmЖ	ЫmЗ	ЬmИ	ЭmЙ	ЮmК	ЯmЛ	†mМ	°mН	ҐmО	£mП	§mР	•mС	¶mТDvУEvФ_vХ`vЦuvЧvvШ	ГvЩ	ДvЪ	РvЫ	СvЬ	ТvЭ	УvЮ	ФvЯ	Хv†	Цv°	ЧvҐ	Шv£	Щv§	Ъv•	Ыv¶	ЬvІ	Эv®	Юv©	Яv™	†vЂ	°vђ	Ґv≠	£vЃ	§vѓ	•v∞	¶v±"
	optimizer
І
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
Ш8
Щ9
Ъ10
Ы11
Ь12
D13
E14
Э15
Ю16
Я17
†18
°19
Ґ20
£21
§22
_23
`24
•25
¶26
u27
v28
Г29
Д30"
trackable_list_wrapper
І
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
Ш8
Щ9
Ъ10
Ы11
Ь12
D13
E14
Э15
Ю16
Я17
†18
°19
Ґ20
£21
§22
_23
`24
•25
¶26
u27
v28
Г29
Д30"
trackable_list_wrapper
 "
trackable_list_wrapper
ѕ
Іnon_trainable_variables
®layers
©metrics
 ™layer_regularization_losses
Ђlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
о2л
(__inference_model_6_layer_call_fn_494776
(__inference_model_6_layer_call_fn_495527
(__inference_model_6_layer_call_fn_495596
(__inference_model_6_layer_call_fn_495272ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
C__inference_model_6_layer_call_and_return_conditional_losses_495810
C__inference_model_6_layer_call_and_return_conditional_losses_496031
C__inference_model_6_layer_call_and_return_conditional_losses_495362
C__inference_model_6_layer_call_and_return_conditional_losses_495452ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЌB 
!__inference__wrapped_model_494181input_17"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
-
ђserving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
√
Рkernel
	Сbias
≠	variables
Ѓtrainable_variables
ѓregularization_losses
∞	keras_api
±__call__
+≤&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
≥	variables
іtrainable_variables
µregularization_losses
ґ	keras_api
Ј__call__
+Є&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Тkernel
	Уbias
є	variables
Їtrainable_variables
їregularization_losses
Љ	keras_api
љ__call__
+Њ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
њ	variables
јtrainable_variables
Ѕregularization_losses
¬	keras_api
√__call__
+ƒ&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Фkernel
	Хbias
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
…__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ќ	keras_api
ѕ__call__
+–&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
P
Р0
С1
Т2
У3
Ф4
Х5"
trackable_list_wrapper
P
Р0
С1
Т2
У3
Ф4
Х5"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
я2№
5__inference_tubelet_embedding_32_layer_call_fn_496119Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jvideos
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъ2ч
P__inference_tubelet_embedding_32_layer_call_and_return_conditional_losses_496155Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jvideos
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
√
Цkernel
	Чbias
№	variables
Ёtrainable_variables
ёregularization_losses
я	keras_api
а__call__
+б&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Шkernel
	Щbias
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Ъkernel
	Ыbias
ф	variables
хtrainable_variables
цregularization_losses
ч	keras_api
ш__call__
+щ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
ъ	variables
ыtrainable_variables
ьregularization_losses
э	keras_api
ю__call__
+€&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"
_tf_keras_layer
P
Ц0
Ч1
Ш2
Щ3
Ъ4
Ы5"
trackable_list_wrapper
P
Ц0
Ч1
Ш2
Щ3
Ъ4
Ы5"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
я2№
5__inference_tubelet_embedding_33_layer_call_fn_496172Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jvideos
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъ2ч
P__inference_tubelet_embedding_33_layer_call_and_return_conditional_losses_496208Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jvideos
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
ў2÷
/__inference_concatenate_13_layer_call_fn_496214Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ф2с
J__inference_concatenate_13_layer_call_and_return_conditional_losses_496221Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Љ
Ь
embeddings
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"
_tf_keras_layer
(
Ь0"
trackable_list_wrapper
(
Ь0"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
и2е
6__inference_positional_encoder_16_layer_call_fn_496230™
°≤Э
FullArgSpec%
argsЪ
jself
jencoded_tokens
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Г2А
Q__inference_positional_encoder_16_layer_call_and_return_conditional_losses_496242™
°≤Э
FullArgSpec%
argsЪ
jself
jencoded_tokens
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
+:)А2layer_normalization_18/gamma
*:(А2layer_normalization_18/beta
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
б2ё
7__inference_layer_normalization_18_layer_call_fn_496251Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ь2щ
R__inference_layer_normalization_18_layer_call_and_return_conditional_losses_496273Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц
†partial_output_shape
°full_output_shape
Эkernel
	Юbias
Ґ	variables
£trainable_variables
§regularization_losses
•	keras_api
¶__call__
+І&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
®partial_output_shape
©full_output_shape
Яkernel
	†bias
™	variables
Ђtrainable_variables
ђregularization_losses
≠	keras_api
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
∞partial_output_shape
±full_output_shape
°kernel
	Ґbias
≤	variables
≥trainable_variables
іregularization_losses
µ	keras_api
ґ__call__
+Ј&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Є	variables
єtrainable_variables
Їregularization_losses
ї	keras_api
Љ__call__
+љ&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Њ	variables
њtrainable_variables
јregularization_losses
Ѕ	keras_api
¬_random_generator
√__call__
+ƒ&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
≈partial_output_shape
∆full_output_shape
£kernel
	§bias
«	variables
»trainable_variables
…regularization_losses
 	keras_api
Ћ__call__
+ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
`
Э0
Ю1
Я2
†3
°4
Ґ5
£6
§7"
trackable_list_wrapper
`
Э0
Ю1
Я2
†3
°4
Ґ5
£6
§7"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
ф2с
7__inference_multi_head_attention_6_layer_call_fn_496297
7__inference_multi_head_attention_6_layer_call_fn_496321ь
у≤п
FullArgSpece
args]ЪZ
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
™2І
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_496357
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_496400ь
у≤п
FullArgSpece
args]ЪZ
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
—2ќ
'__inference_add_12_layer_call_fn_496406Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_add_12_layer_call_and_return_conditional_losses_496412Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
+:)А2layer_normalization_19/gamma
*:(А2layer_normalization_19/beta
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
б2ё
7__inference_layer_normalization_19_layer_call_fn_496421Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ь2щ
R__inference_layer_normalization_19_layer_call_and_return_conditional_losses_496443Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
√
•kernel
	¶bias
№	variables
Ёtrainable_variables
ёregularization_losses
я	keras_api
а__call__
+б&call_and_return_all_conditional_losses"
_tf_keras_layer
0
•0
¶1"
trackable_list_wrapper
0
•0
¶1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
В2€
-__inference_sequential_6_layer_call_fn_494312
-__inference_sequential_6_layer_call_fn_496452
-__inference_sequential_6_layer_call_fn_496461
-__inference_sequential_6_layer_call_fn_494358ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
H__inference_sequential_6_layer_call_and_return_conditional_losses_496499
H__inference_sequential_6_layer_call_and_return_conditional_losses_496537
H__inference_sequential_6_layer_call_and_return_conditional_losses_494367
H__inference_sequential_6_layer_call_and_return_conditional_losses_494376ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
—2ќ
'__inference_add_13_layer_call_fn_496543Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_add_13_layer_call_and_return_conditional_losses_496549Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
+:)А2layer_normalization_20/gamma
*:(А2layer_normalization_20/beta
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
б2ё
7__inference_layer_normalization_20_layer_call_fn_496558Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ь2щ
R__inference_layer_normalization_20_layer_call_and_return_conditional_losses_496580Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
}	variables
~trainable_variables
regularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
т2п
;__inference_global_average_pooling1d_6_layer_call_fn_496585ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Н2К
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_496591ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
": 	А2dense_13/kernel
:2dense_13/bias
0
Г0
Д1"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
”2–
)__inference_dense_13_layer_call_fn_496600Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_13_layer_call_and_return_conditional_losses_496610Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
C:A 2%tubelet_embedding_32/conv3d_96/kernel
1:/ 2#tubelet_embedding_32/conv3d_96/bias
C:A @2%tubelet_embedding_32/conv3d_97/kernel
1:/@2#tubelet_embedding_32/conv3d_97/bias
D:B@А2%tubelet_embedding_32/conv3d_98/kernel
2:0А2#tubelet_embedding_32/conv3d_98/bias
C:A 2%tubelet_embedding_33/conv3d_99/kernel
1:/ 2#tubelet_embedding_33/conv3d_99/bias
D:B @2&tubelet_embedding_33/conv3d_100/kernel
2:0@2$tubelet_embedding_33/conv3d_100/bias
E:C@А2&tubelet_embedding_33/conv3d_101/kernel
3:1А2$tubelet_embedding_33/conv3d_101/bias
=:;	(А2*positional_encoder_16/embedding/embeddings
;:9АА2#multi_head_attention_6/query/kernel
4:2	А2!multi_head_attention_6/query/bias
9:7АА2!multi_head_attention_6/key/kernel
2:0	А2multi_head_attention_6/key/bias
;:9АА2#multi_head_attention_6/value/kernel
4:2	А2!multi_head_attention_6/value/bias
F:DАА2.multi_head_attention_6/attention_output/kernel
;:9А2,multi_head_attention_6/attention_output/bias
#:!
АА2dense_12/kernel
:А2dense_12/bias
 "
trackable_list_wrapper
Ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
0
ы0
ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ћB…
$__inference_signature_wrapper_496102input_17"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Р0
С1"
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
≠	variables
Ѓtrainable_variables
ѓregularization_losses
±__call__
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
≥	variables
іtrainable_variables
µregularization_losses
Ј__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
џ2Ў
1__inference_max_pooling3d_64_layer_call_fn_496615Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц2у
L__inference_max_pooling3d_64_layer_call_and_return_conditional_losses_496620Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Т0
У1"
trackable_list_wrapper
0
Т0
У1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
є	variables
Їtrainable_variables
їregularization_losses
љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
њ	variables
јtrainable_variables
Ѕregularization_losses
√__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses"
_generic_user_object
џ2Ў
1__inference_max_pooling3d_65_layer_call_fn_496625Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц2у
L__inference_max_pooling3d_65_layer_call_and_return_conditional_losses_496630Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Ф0
Х1"
trackable_list_wrapper
0
Ф0
Х1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
≈	variables
∆trainable_variables
«regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ѕ__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses"
_generic_user_object
я2№
5__inference_average_pooling3d_32_layer_call_fn_496635Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъ2ч
P__inference_average_pooling3d_32_layer_call_and_return_conditional_losses_496640Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
Q
0
1
2
3
 4
!5
"6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ц0
Ч1"
trackable_list_wrapper
0
Ц0
Ч1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
№	variables
Ёtrainable_variables
ёregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
џ2Ў
1__inference_max_pooling3d_66_layer_call_fn_496645Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц2у
L__inference_max_pooling3d_66_layer_call_and_return_conditional_losses_496650Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Ш0
Щ1"
trackable_list_wrapper
0
Ш0
Щ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
™non_trainable_variables
Ђlayers
ђmetrics
 ≠layer_regularization_losses
Ѓlayer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ѓnon_trainable_variables
∞layers
±metrics
 ≤layer_regularization_losses
≥layer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
џ2Ў
1__inference_max_pooling3d_67_layer_call_fn_496655Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц2у
L__inference_max_pooling3d_67_layer_call_and_return_conditional_losses_496660Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Ъ0
Ы1"
trackable_list_wrapper
0
Ъ0
Ы1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
ф	variables
хtrainable_variables
цregularization_losses
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
ъ	variables
ыtrainable_variables
ьregularization_losses
ю__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
я2№
5__inference_average_pooling3d_33_layer_call_fn_496665Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъ2ч
P__inference_average_pooling3d_33_layer_call_and_return_conditional_losses_496670Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
Q
)0
*1
+2
,3
-4
.5
/6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
Ь0"
trackable_list_wrapper
(
Ь0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Э0
Ю1"
trackable_list_wrapper
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
Ґ	variables
£trainable_variables
§regularization_losses
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Я0
†1"
trackable_list_wrapper
0
Я0
†1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
™	variables
Ђtrainable_variables
ђregularization_losses
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
°0
Ґ1"
trackable_list_wrapper
0
°0
Ґ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
≤	variables
≥trainable_variables
іregularization_losses
ґ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
Є	variables
єtrainable_variables
Їregularization_losses
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
№non_trainable_variables
Ёlayers
ёmetrics
 яlayer_regularization_losses
аlayer_metrics
Њ	variables
њtrainable_variables
јregularization_losses
√__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
£0
§1"
trackable_list_wrapper
0
£0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
«	variables
»trainable_variables
…regularization_losses
Ћ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
J
L0
M1
N2
O3
P4
Q5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
•0
¶1"
trackable_list_wrapper
0
•0
¶1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
№	variables
Ёtrainable_variables
ёregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
”2–
)__inference_dense_12_layer_call_fn_496679Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_12_layer_call_and_return_conditional_losses_496717Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

лtotal

мcount
н	variables
о	keras_api"
_tf_keras_metric
c

пtotal

рcount
с
_fn_kwargs
т	variables
у	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
л0
м1"
trackable_list_wrapper
.
н	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
п0
р1"
trackable_list_wrapper
.
т	variables"
_generic_user_object
0:.А2#Adam/layer_normalization_18/gamma/m
/:-А2"Adam/layer_normalization_18/beta/m
0:.А2#Adam/layer_normalization_19/gamma/m
/:-А2"Adam/layer_normalization_19/beta/m
0:.А2#Adam/layer_normalization_20/gamma/m
/:-А2"Adam/layer_normalization_20/beta/m
':%	А2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
H:F 2,Adam/tubelet_embedding_32/conv3d_96/kernel/m
6:4 2*Adam/tubelet_embedding_32/conv3d_96/bias/m
H:F @2,Adam/tubelet_embedding_32/conv3d_97/kernel/m
6:4@2*Adam/tubelet_embedding_32/conv3d_97/bias/m
I:G@А2,Adam/tubelet_embedding_32/conv3d_98/kernel/m
7:5А2*Adam/tubelet_embedding_32/conv3d_98/bias/m
H:F 2,Adam/tubelet_embedding_33/conv3d_99/kernel/m
6:4 2*Adam/tubelet_embedding_33/conv3d_99/bias/m
I:G @2-Adam/tubelet_embedding_33/conv3d_100/kernel/m
7:5@2+Adam/tubelet_embedding_33/conv3d_100/bias/m
J:H@А2-Adam/tubelet_embedding_33/conv3d_101/kernel/m
8:6А2+Adam/tubelet_embedding_33/conv3d_101/bias/m
B:@	(А21Adam/positional_encoder_16/embedding/embeddings/m
@:>АА2*Adam/multi_head_attention_6/query/kernel/m
9:7	А2(Adam/multi_head_attention_6/query/bias/m
>:<АА2(Adam/multi_head_attention_6/key/kernel/m
7:5	А2&Adam/multi_head_attention_6/key/bias/m
@:>АА2*Adam/multi_head_attention_6/value/kernel/m
9:7	А2(Adam/multi_head_attention_6/value/bias/m
K:IАА25Adam/multi_head_attention_6/attention_output/kernel/m
@:>А23Adam/multi_head_attention_6/attention_output/bias/m
(:&
АА2Adam/dense_12/kernel/m
!:А2Adam/dense_12/bias/m
0:.А2#Adam/layer_normalization_18/gamma/v
/:-А2"Adam/layer_normalization_18/beta/v
0:.А2#Adam/layer_normalization_19/gamma/v
/:-А2"Adam/layer_normalization_19/beta/v
0:.А2#Adam/layer_normalization_20/gamma/v
/:-А2"Adam/layer_normalization_20/beta/v
':%	А2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
H:F 2,Adam/tubelet_embedding_32/conv3d_96/kernel/v
6:4 2*Adam/tubelet_embedding_32/conv3d_96/bias/v
H:F @2,Adam/tubelet_embedding_32/conv3d_97/kernel/v
6:4@2*Adam/tubelet_embedding_32/conv3d_97/bias/v
I:G@А2,Adam/tubelet_embedding_32/conv3d_98/kernel/v
7:5А2*Adam/tubelet_embedding_32/conv3d_98/bias/v
H:F 2,Adam/tubelet_embedding_33/conv3d_99/kernel/v
6:4 2*Adam/tubelet_embedding_33/conv3d_99/bias/v
I:G @2-Adam/tubelet_embedding_33/conv3d_100/kernel/v
7:5@2+Adam/tubelet_embedding_33/conv3d_100/bias/v
J:H@А2-Adam/tubelet_embedding_33/conv3d_101/kernel/v
8:6А2+Adam/tubelet_embedding_33/conv3d_101/bias/v
B:@	(А21Adam/positional_encoder_16/embedding/embeddings/v
@:>АА2*Adam/multi_head_attention_6/query/kernel/v
9:7	А2(Adam/multi_head_attention_6/query/bias/v
>:<АА2(Adam/multi_head_attention_6/key/kernel/v
7:5	А2&Adam/multi_head_attention_6/key/bias/v
@:>АА2*Adam/multi_head_attention_6/value/kernel/v
9:7	А2(Adam/multi_head_attention_6/value/bias/v
K:IАА25Adam/multi_head_attention_6/attention_output/kernel/v
@:>А23Adam/multi_head_attention_6/attention_output/bias/v
(:&
АА2Adam/dense_12/kernel/v
!:А2Adam/dense_12/bias/v
	J
Const÷
!__inference__wrapped_model_494181∞:РСТУФХЦЧШЩЪЫ≤ЬDEЭЮЯ†°Ґ£§_`•¶uvГД=Ґ:
3Ґ0
.К+
input_17€€€€€€€€€

™ "3™0
.
dense_13"К
dense_13€€€€€€€€€ў
B__inference_add_12_layer_call_and_return_conditional_losses_496412ТdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€(А
'К$
inputs/1€€€€€€€€€(А
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ ±
'__inference_add_12_layer_call_fn_496406ЕdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€(А
'К$
inputs/1€€€€€€€€€(А
™ "К€€€€€€€€€(Аў
B__inference_add_13_layer_call_and_return_conditional_losses_496549ТdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€(А
'К$
inputs/1€€€€€€€€€(А
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ ±
'__inference_add_13_layer_call_fn_496543ЕdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€(А
'К$
inputs/1€€€€€€€€€(А
™ "К€€€€€€€€€(АН
P__inference_average_pooling3d_32_layer_call_and_return_conditional_losses_496640Є_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "UҐR
KКH
0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ е
5__inference_average_pooling3d_32_layer_call_fn_496635Ђ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HКEA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
P__inference_average_pooling3d_33_layer_call_and_return_conditional_losses_496670Є_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "UҐR
KКH
0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ е
5__inference_average_pooling3d_33_layer_call_fn_496665Ђ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HКEA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€б
J__inference_concatenate_13_layer_call_and_return_conditional_losses_496221ТdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€(А
'К$
inputs/1€€€€€€€€€(А
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ є
/__inference_concatenate_13_layer_call_fn_496214ЕdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€(А
'К$
inputs/1€€€€€€€€€(А
™ "К€€€€€€€€€(А∞
D__inference_dense_12_layer_call_and_return_conditional_losses_496717h•¶4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€(А
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ И
)__inference_dense_12_layer_call_fn_496679[•¶4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€(А
™ "К€€€€€€€€€(АІ
D__inference_dense_13_layer_call_and_return_conditional_losses_496610_ГД0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ 
)__inference_dense_13_layer_call_fn_496600RГД0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€’
V__inference_global_average_pooling1d_6_layer_call_and_return_conditional_losses_496591{IҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ ≠
;__inference_global_average_pooling1d_6_layer_call_fn_496585nIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "!К€€€€€€€€€€€€€€€€€€Љ
R__inference_layer_normalization_18_layer_call_and_return_conditional_losses_496273fDE4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€(А
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ Ф
7__inference_layer_normalization_18_layer_call_fn_496251YDE4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€(А
™ "К€€€€€€€€€(АЉ
R__inference_layer_normalization_19_layer_call_and_return_conditional_losses_496443f_`4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€(А
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ Ф
7__inference_layer_normalization_19_layer_call_fn_496421Y_`4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€(А
™ "К€€€€€€€€€(АЉ
R__inference_layer_normalization_20_layer_call_and_return_conditional_losses_496580fuv4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€(А
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ Ф
7__inference_layer_normalization_20_layer_call_fn_496558Yuv4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€(А
™ "К€€€€€€€€€(АЙ
L__inference_max_pooling3d_64_layer_call_and_return_conditional_losses_496620Є_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "UҐR
KКH
0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ б
1__inference_max_pooling3d_64_layer_call_fn_496615Ђ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HКEA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Й
L__inference_max_pooling3d_65_layer_call_and_return_conditional_losses_496630Є_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "UҐR
KКH
0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ б
1__inference_max_pooling3d_65_layer_call_fn_496625Ђ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HКEA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Й
L__inference_max_pooling3d_66_layer_call_and_return_conditional_losses_496650Є_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "UҐR
KКH
0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ б
1__inference_max_pooling3d_66_layer_call_fn_496645Ђ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HКEA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Й
L__inference_max_pooling3d_67_layer_call_and_return_conditional_losses_496660Є_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "UҐR
KКH
0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ б
1__inference_max_pooling3d_67_layer_call_fn_496655Ђ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HКEA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€т
C__inference_model_6_layer_call_and_return_conditional_losses_495362™:РСТУФХЦЧШЩЪЫ≤ЬDEЭЮЯ†°Ґ£§_`•¶uvГДEҐB
;Ґ8
.К+
input_17€€€€€€€€€

p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ т
C__inference_model_6_layer_call_and_return_conditional_losses_495452™:РСТУФХЦЧШЩЪЫ≤ЬDEЭЮЯ†°Ґ£§_`•¶uvГДEҐB
;Ґ8
.К+
input_17€€€€€€€€€

p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ р
C__inference_model_6_layer_call_and_return_conditional_losses_495810®:РСТУФХЦЧШЩЪЫ≤ЬDEЭЮЯ†°Ґ£§_`•¶uvГДCҐ@
9Ґ6
,К)
inputs€€€€€€€€€

p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ р
C__inference_model_6_layer_call_and_return_conditional_losses_496031®:РСТУФХЦЧШЩЪЫ≤ЬDEЭЮЯ†°Ґ£§_`•¶uvГДCҐ@
9Ґ6
,К)
inputs€€€€€€€€€

p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ  
(__inference_model_6_layer_call_fn_494776Э:РСТУФХЦЧШЩЪЫ≤ЬDEЭЮЯ†°Ґ£§_`•¶uvГДEҐB
;Ґ8
.К+
input_17€€€€€€€€€

p 

 
™ "К€€€€€€€€€ 
(__inference_model_6_layer_call_fn_495272Э:РСТУФХЦЧШЩЪЫ≤ЬDEЭЮЯ†°Ґ£§_`•¶uvГДEҐB
;Ґ8
.К+
input_17€€€€€€€€€

p

 
™ "К€€€€€€€€€»
(__inference_model_6_layer_call_fn_495527Ы:РСТУФХЦЧШЩЪЫ≤ЬDEЭЮЯ†°Ґ£§_`•¶uvГДCҐ@
9Ґ6
,К)
inputs€€€€€€€€€

p 

 
™ "К€€€€€€€€€»
(__inference_model_6_layer_call_fn_495596Ы:РСТУФХЦЧШЩЪЫ≤ЬDEЭЮЯ†°Ґ£§_`•¶uvГДCҐ@
9Ґ6
,К)
inputs€€€€€€€€€

p

 
™ "К€€€€€€€€€Ѓ
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_496357„ЭЮЯ†°Ґ£§iҐf
_Ґ\
$К!
query€€€€€€€€€(А
$К!
value€€€€€€€€€(А

 

 
p
p 
™ "XҐU
NҐK
"К
0/0€€€€€€€€€(А
%К"
0/1€€€€€€€€€((
Ъ Ѓ
R__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_496400„ЭЮЯ†°Ґ£§iҐf
_Ґ\
$К!
query€€€€€€€€€(А
$К!
value€€€€€€€€€(А

 

 
p
p
™ "XҐU
NҐK
"К
0/0€€€€€€€€€(А
%К"
0/1€€€€€€€€€((
Ъ Е
7__inference_multi_head_attention_6_layer_call_fn_496297…ЭЮЯ†°Ґ£§iҐf
_Ґ\
$К!
query€€€€€€€€€(А
$К!
value€€€€€€€€€(А

 

 
p
p 
™ "JҐG
 К
0€€€€€€€€€(А
#К 
1€€€€€€€€€((Е
7__inference_multi_head_attention_6_layer_call_fn_496321…ЭЮЯ†°Ґ£§iҐf
_Ґ\
$К!
query€€€€€€€€€(А
$К!
value€€€€€€€€€(А

 

 
p
p
™ "JҐG
 К
0€€€€€€€€€(А
#К 
1€€€€€€€€€((≈
Q__inference_positional_encoder_16_layer_call_and_return_conditional_losses_496242p≤Ь<Ґ9
2Ґ/
-К*
encoded_tokens€€€€€€€€€(А
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ Э
6__inference_positional_encoder_16_layer_call_fn_496230c≤Ь<Ґ9
2Ґ/
-К*
encoded_tokens€€€€€€€€€(А
™ "К€€€€€€€€€(Аƒ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494367x•¶DҐA
:Ґ7
-К*
dense_12_input€€€€€€€€€(А
p 

 
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ ƒ
H__inference_sequential_6_layer_call_and_return_conditional_losses_494376x•¶DҐA
:Ґ7
-К*
dense_12_input€€€€€€€€€(А
p

 
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ Љ
H__inference_sequential_6_layer_call_and_return_conditional_losses_496499p•¶<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€(А
p 

 
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ Љ
H__inference_sequential_6_layer_call_and_return_conditional_losses_496537p•¶<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€(А
p

 
™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ Ь
-__inference_sequential_6_layer_call_fn_494312k•¶DҐA
:Ґ7
-К*
dense_12_input€€€€€€€€€(А
p 

 
™ "К€€€€€€€€€(АЬ
-__inference_sequential_6_layer_call_fn_494358k•¶DҐA
:Ґ7
-К*
dense_12_input€€€€€€€€€(А
p

 
™ "К€€€€€€€€€(АФ
-__inference_sequential_6_layer_call_fn_496452c•¶<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€(А
p 

 
™ "К€€€€€€€€€(АФ
-__inference_sequential_6_layer_call_fn_496461c•¶<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€(А
p

 
™ "К€€€€€€€€€(Ае
$__inference_signature_wrapper_496102Љ:РСТУФХЦЧШЩЪЫ≤ЬDEЭЮЯ†°Ґ£§_`•¶uvГДIҐF
Ґ 
?™<
:
input_17.К+
input_17€€€€€€€€€
"3™0
.
dense_13"К
dense_13€€€€€€€€€Ћ
P__inference_tubelet_embedding_32_layer_call_and_return_conditional_losses_496155wРСТУФХ;Ґ8
1Ґ.
,К)
videos€€€€€€€€€

™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ £
5__inference_tubelet_embedding_32_layer_call_fn_496119jРСТУФХ;Ґ8
1Ґ.
,К)
videos€€€€€€€€€

™ "К€€€€€€€€€(АЋ
P__inference_tubelet_embedding_33_layer_call_and_return_conditional_losses_496208wЦЧШЩЪЫ;Ґ8
1Ґ.
,К)
videos€€€€€€€€€

™ "*Ґ'
 К
0€€€€€€€€€(А
Ъ £
5__inference_tubelet_embedding_33_layer_call_fn_496172jЦЧШЩЪЫ;Ґ8
1Ґ.
,К)
videos€€€€€€€€€

™ "К€€€€€€€€€(А