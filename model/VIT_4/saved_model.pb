��!
� � 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
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
�
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
�
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
�
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
�
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
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
�
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
dtypetype�
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
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
list(type)(0�
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

2	�
�
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
executor_typestring ��
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
�
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��
�
layer_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namelayer_normalization_15/gamma
�
0layer_normalization_15/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_15/gamma*
_output_shapes	
:�*
dtype0
�
layer_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namelayer_normalization_15/beta
�
/layer_normalization_15/beta/Read/ReadVariableOpReadVariableOplayer_normalization_15/beta*
_output_shapes	
:�*
dtype0
�
layer_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namelayer_normalization_16/gamma
�
0layer_normalization_16/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_16/gamma*
_output_shapes	
:�*
dtype0
�
layer_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namelayer_normalization_16/beta
�
/layer_normalization_16/beta/Read/ReadVariableOpReadVariableOplayer_normalization_16/beta*
_output_shapes	
:�*
dtype0
�
layer_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namelayer_normalization_17/gamma
�
0layer_normalization_17/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_17/gamma*
_output_shapes	
:�*
dtype0
�
layer_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namelayer_normalization_17/beta
�
/layer_normalization_17/beta/Read/ReadVariableOpReadVariableOplayer_normalization_17/beta*
_output_shapes	
:�*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	�*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
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
�
%tubelet_embedding_30/conv3d_90/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%tubelet_embedding_30/conv3d_90/kernel
�
9tubelet_embedding_30/conv3d_90/kernel/Read/ReadVariableOpReadVariableOp%tubelet_embedding_30/conv3d_90/kernel**
_output_shapes
: *
dtype0
�
#tubelet_embedding_30/conv3d_90/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#tubelet_embedding_30/conv3d_90/bias
�
7tubelet_embedding_30/conv3d_90/bias/Read/ReadVariableOpReadVariableOp#tubelet_embedding_30/conv3d_90/bias*
_output_shapes
: *
dtype0
�
%tubelet_embedding_30/conv3d_91/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%tubelet_embedding_30/conv3d_91/kernel
�
9tubelet_embedding_30/conv3d_91/kernel/Read/ReadVariableOpReadVariableOp%tubelet_embedding_30/conv3d_91/kernel**
_output_shapes
: @*
dtype0
�
#tubelet_embedding_30/conv3d_91/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#tubelet_embedding_30/conv3d_91/bias
�
7tubelet_embedding_30/conv3d_91/bias/Read/ReadVariableOpReadVariableOp#tubelet_embedding_30/conv3d_91/bias*
_output_shapes
:@*
dtype0
�
%tubelet_embedding_30/conv3d_92/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@�*6
shared_name'%tubelet_embedding_30/conv3d_92/kernel
�
9tubelet_embedding_30/conv3d_92/kernel/Read/ReadVariableOpReadVariableOp%tubelet_embedding_30/conv3d_92/kernel*+
_output_shapes
:@�*
dtype0
�
#tubelet_embedding_30/conv3d_92/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#tubelet_embedding_30/conv3d_92/bias
�
7tubelet_embedding_30/conv3d_92/bias/Read/ReadVariableOpReadVariableOp#tubelet_embedding_30/conv3d_92/bias*
_output_shapes	
:�*
dtype0
�
%tubelet_embedding_31/conv3d_93/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%tubelet_embedding_31/conv3d_93/kernel
�
9tubelet_embedding_31/conv3d_93/kernel/Read/ReadVariableOpReadVariableOp%tubelet_embedding_31/conv3d_93/kernel**
_output_shapes
: *
dtype0
�
#tubelet_embedding_31/conv3d_93/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#tubelet_embedding_31/conv3d_93/bias
�
7tubelet_embedding_31/conv3d_93/bias/Read/ReadVariableOpReadVariableOp#tubelet_embedding_31/conv3d_93/bias*
_output_shapes
: *
dtype0
�
%tubelet_embedding_31/conv3d_94/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%tubelet_embedding_31/conv3d_94/kernel
�
9tubelet_embedding_31/conv3d_94/kernel/Read/ReadVariableOpReadVariableOp%tubelet_embedding_31/conv3d_94/kernel**
_output_shapes
: @*
dtype0
�
#tubelet_embedding_31/conv3d_94/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#tubelet_embedding_31/conv3d_94/bias
�
7tubelet_embedding_31/conv3d_94/bias/Read/ReadVariableOpReadVariableOp#tubelet_embedding_31/conv3d_94/bias*
_output_shapes
:@*
dtype0
�
%tubelet_embedding_31/conv3d_95/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@�*6
shared_name'%tubelet_embedding_31/conv3d_95/kernel
�
9tubelet_embedding_31/conv3d_95/kernel/Read/ReadVariableOpReadVariableOp%tubelet_embedding_31/conv3d_95/kernel*+
_output_shapes
:@�*
dtype0
�
#tubelet_embedding_31/conv3d_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#tubelet_embedding_31/conv3d_95/bias
�
7tubelet_embedding_31/conv3d_95/bias/Read/ReadVariableOpReadVariableOp#tubelet_embedding_31/conv3d_95/bias*
_output_shapes	
:�*
dtype0
�
*positional_encoder_15/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(�*;
shared_name,*positional_encoder_15/embedding/embeddings
�
>positional_encoder_15/embedding/embeddings/Read/ReadVariableOpReadVariableOp*positional_encoder_15/embedding/embeddings*
_output_shapes
:	(�*
dtype0
�
#multi_head_attention_5/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*4
shared_name%#multi_head_attention_5/query/kernel
�
7multi_head_attention_5/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_5/query/kernel*$
_output_shapes
:��*
dtype0
�
!multi_head_attention_5/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*2
shared_name#!multi_head_attention_5/query/bias
�
5multi_head_attention_5/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_5/query/bias*
_output_shapes
:	�*
dtype0
�
!multi_head_attention_5/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*2
shared_name#!multi_head_attention_5/key/kernel
�
5multi_head_attention_5/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_5/key/kernel*$
_output_shapes
:��*
dtype0
�
multi_head_attention_5/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*0
shared_name!multi_head_attention_5/key/bias
�
3multi_head_attention_5/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_5/key/bias*
_output_shapes
:	�*
dtype0
�
#multi_head_attention_5/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*4
shared_name%#multi_head_attention_5/value/kernel
�
7multi_head_attention_5/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_5/value/kernel*$
_output_shapes
:��*
dtype0
�
!multi_head_attention_5/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*2
shared_name#!multi_head_attention_5/value/bias
�
5multi_head_attention_5/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_5/value/bias*
_output_shapes
:	�*
dtype0
�
.multi_head_attention_5/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*?
shared_name0.multi_head_attention_5/attention_output/kernel
�
Bmulti_head_attention_5/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_5/attention_output/kernel*$
_output_shapes
:��*
dtype0
�
,multi_head_attention_5/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,multi_head_attention_5/attention_output/bias
�
@multi_head_attention_5/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_5/attention_output/bias*
_output_shapes	
:�*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
��*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:�*
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
�
#Adam/layer_normalization_15/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/layer_normalization_15/gamma/m
�
7Adam/layer_normalization_15/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_15/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/layer_normalization_15/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/layer_normalization_15/beta/m
�
6Adam/layer_normalization_15/beta/m/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_15/beta/m*
_output_shapes	
:�*
dtype0
�
#Adam/layer_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/layer_normalization_16/gamma/m
�
7Adam/layer_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_16/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/layer_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/layer_normalization_16/beta/m
�
6Adam/layer_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_16/beta/m*
_output_shapes	
:�*
dtype0
�
#Adam/layer_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/layer_normalization_17/gamma/m
�
7Adam/layer_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_17/gamma/m*
_output_shapes	
:�*
dtype0
�
"Adam/layer_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/layer_normalization_17/beta/m
�
6Adam/layer_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_17/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_11/kernel/m
�
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:*
dtype0
�
,Adam/tubelet_embedding_30/conv3d_90/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/tubelet_embedding_30/conv3d_90/kernel/m
�
@Adam/tubelet_embedding_30/conv3d_90/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_30/conv3d_90/kernel/m**
_output_shapes
: *
dtype0
�
*Adam/tubelet_embedding_30/conv3d_90/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/tubelet_embedding_30/conv3d_90/bias/m
�
>Adam/tubelet_embedding_30/conv3d_90/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_30/conv3d_90/bias/m*
_output_shapes
: *
dtype0
�
,Adam/tubelet_embedding_30/conv3d_91/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/tubelet_embedding_30/conv3d_91/kernel/m
�
@Adam/tubelet_embedding_30/conv3d_91/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_30/conv3d_91/kernel/m**
_output_shapes
: @*
dtype0
�
*Adam/tubelet_embedding_30/conv3d_91/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/tubelet_embedding_30/conv3d_91/bias/m
�
>Adam/tubelet_embedding_30/conv3d_91/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_30/conv3d_91/bias/m*
_output_shapes
:@*
dtype0
�
,Adam/tubelet_embedding_30/conv3d_92/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@�*=
shared_name.,Adam/tubelet_embedding_30/conv3d_92/kernel/m
�
@Adam/tubelet_embedding_30/conv3d_92/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_30/conv3d_92/kernel/m*+
_output_shapes
:@�*
dtype0
�
*Adam/tubelet_embedding_30/conv3d_92/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*Adam/tubelet_embedding_30/conv3d_92/bias/m
�
>Adam/tubelet_embedding_30/conv3d_92/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_30/conv3d_92/bias/m*
_output_shapes	
:�*
dtype0
�
,Adam/tubelet_embedding_31/conv3d_93/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/tubelet_embedding_31/conv3d_93/kernel/m
�
@Adam/tubelet_embedding_31/conv3d_93/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_31/conv3d_93/kernel/m**
_output_shapes
: *
dtype0
�
*Adam/tubelet_embedding_31/conv3d_93/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/tubelet_embedding_31/conv3d_93/bias/m
�
>Adam/tubelet_embedding_31/conv3d_93/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_31/conv3d_93/bias/m*
_output_shapes
: *
dtype0
�
,Adam/tubelet_embedding_31/conv3d_94/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/tubelet_embedding_31/conv3d_94/kernel/m
�
@Adam/tubelet_embedding_31/conv3d_94/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_31/conv3d_94/kernel/m**
_output_shapes
: @*
dtype0
�
*Adam/tubelet_embedding_31/conv3d_94/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/tubelet_embedding_31/conv3d_94/bias/m
�
>Adam/tubelet_embedding_31/conv3d_94/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_31/conv3d_94/bias/m*
_output_shapes
:@*
dtype0
�
,Adam/tubelet_embedding_31/conv3d_95/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@�*=
shared_name.,Adam/tubelet_embedding_31/conv3d_95/kernel/m
�
@Adam/tubelet_embedding_31/conv3d_95/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_31/conv3d_95/kernel/m*+
_output_shapes
:@�*
dtype0
�
*Adam/tubelet_embedding_31/conv3d_95/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*Adam/tubelet_embedding_31/conv3d_95/bias/m
�
>Adam/tubelet_embedding_31/conv3d_95/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_31/conv3d_95/bias/m*
_output_shapes	
:�*
dtype0
�
1Adam/positional_encoder_15/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(�*B
shared_name31Adam/positional_encoder_15/embedding/embeddings/m
�
EAdam/positional_encoder_15/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp1Adam/positional_encoder_15/embedding/embeddings/m*
_output_shapes
:	(�*
dtype0
�
*Adam/multi_head_attention_5/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*;
shared_name,*Adam/multi_head_attention_5/query/kernel/m
�
>Adam/multi_head_attention_5/query/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_5/query/kernel/m*$
_output_shapes
:��*
dtype0
�
(Adam/multi_head_attention_5/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*9
shared_name*(Adam/multi_head_attention_5/query/bias/m
�
<Adam/multi_head_attention_5/query/bias/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_5/query/bias/m*
_output_shapes
:	�*
dtype0
�
(Adam/multi_head_attention_5/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*9
shared_name*(Adam/multi_head_attention_5/key/kernel/m
�
<Adam/multi_head_attention_5/key/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_5/key/kernel/m*$
_output_shapes
:��*
dtype0
�
&Adam/multi_head_attention_5/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*7
shared_name(&Adam/multi_head_attention_5/key/bias/m
�
:Adam/multi_head_attention_5/key/bias/m/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention_5/key/bias/m*
_output_shapes
:	�*
dtype0
�
*Adam/multi_head_attention_5/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*;
shared_name,*Adam/multi_head_attention_5/value/kernel/m
�
>Adam/multi_head_attention_5/value/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_5/value/kernel/m*$
_output_shapes
:��*
dtype0
�
(Adam/multi_head_attention_5/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*9
shared_name*(Adam/multi_head_attention_5/value/bias/m
�
<Adam/multi_head_attention_5/value/bias/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_5/value/bias/m*
_output_shapes
:	�*
dtype0
�
5Adam/multi_head_attention_5/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*F
shared_name75Adam/multi_head_attention_5/attention_output/kernel/m
�
IAdam/multi_head_attention_5/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/multi_head_attention_5/attention_output/kernel/m*$
_output_shapes
:��*
dtype0
�
3Adam/multi_head_attention_5/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53Adam/multi_head_attention_5/attention_output/bias/m
�
GAdam/multi_head_attention_5/attention_output/bias/m/Read/ReadVariableOpReadVariableOp3Adam/multi_head_attention_5/attention_output/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_10/kernel/m
�
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_10/bias/m
z
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/layer_normalization_15/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/layer_normalization_15/gamma/v
�
7Adam/layer_normalization_15/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_15/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/layer_normalization_15/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/layer_normalization_15/beta/v
�
6Adam/layer_normalization_15/beta/v/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_15/beta/v*
_output_shapes	
:�*
dtype0
�
#Adam/layer_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/layer_normalization_16/gamma/v
�
7Adam/layer_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_16/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/layer_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/layer_normalization_16/beta/v
�
6Adam/layer_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_16/beta/v*
_output_shapes	
:�*
dtype0
�
#Adam/layer_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/layer_normalization_17/gamma/v
�
7Adam/layer_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_17/gamma/v*
_output_shapes	
:�*
dtype0
�
"Adam/layer_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/layer_normalization_17/beta/v
�
6Adam/layer_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_17/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_11/kernel/v
�
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:*
dtype0
�
,Adam/tubelet_embedding_30/conv3d_90/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/tubelet_embedding_30/conv3d_90/kernel/v
�
@Adam/tubelet_embedding_30/conv3d_90/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_30/conv3d_90/kernel/v**
_output_shapes
: *
dtype0
�
*Adam/tubelet_embedding_30/conv3d_90/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/tubelet_embedding_30/conv3d_90/bias/v
�
>Adam/tubelet_embedding_30/conv3d_90/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_30/conv3d_90/bias/v*
_output_shapes
: *
dtype0
�
,Adam/tubelet_embedding_30/conv3d_91/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/tubelet_embedding_30/conv3d_91/kernel/v
�
@Adam/tubelet_embedding_30/conv3d_91/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_30/conv3d_91/kernel/v**
_output_shapes
: @*
dtype0
�
*Adam/tubelet_embedding_30/conv3d_91/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/tubelet_embedding_30/conv3d_91/bias/v
�
>Adam/tubelet_embedding_30/conv3d_91/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_30/conv3d_91/bias/v*
_output_shapes
:@*
dtype0
�
,Adam/tubelet_embedding_30/conv3d_92/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@�*=
shared_name.,Adam/tubelet_embedding_30/conv3d_92/kernel/v
�
@Adam/tubelet_embedding_30/conv3d_92/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_30/conv3d_92/kernel/v*+
_output_shapes
:@�*
dtype0
�
*Adam/tubelet_embedding_30/conv3d_92/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*Adam/tubelet_embedding_30/conv3d_92/bias/v
�
>Adam/tubelet_embedding_30/conv3d_92/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_30/conv3d_92/bias/v*
_output_shapes	
:�*
dtype0
�
,Adam/tubelet_embedding_31/conv3d_93/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/tubelet_embedding_31/conv3d_93/kernel/v
�
@Adam/tubelet_embedding_31/conv3d_93/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_31/conv3d_93/kernel/v**
_output_shapes
: *
dtype0
�
*Adam/tubelet_embedding_31/conv3d_93/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/tubelet_embedding_31/conv3d_93/bias/v
�
>Adam/tubelet_embedding_31/conv3d_93/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_31/conv3d_93/bias/v*
_output_shapes
: *
dtype0
�
,Adam/tubelet_embedding_31/conv3d_94/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/tubelet_embedding_31/conv3d_94/kernel/v
�
@Adam/tubelet_embedding_31/conv3d_94/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_31/conv3d_94/kernel/v**
_output_shapes
: @*
dtype0
�
*Adam/tubelet_embedding_31/conv3d_94/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/tubelet_embedding_31/conv3d_94/bias/v
�
>Adam/tubelet_embedding_31/conv3d_94/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_31/conv3d_94/bias/v*
_output_shapes
:@*
dtype0
�
,Adam/tubelet_embedding_31/conv3d_95/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@�*=
shared_name.,Adam/tubelet_embedding_31/conv3d_95/kernel/v
�
@Adam/tubelet_embedding_31/conv3d_95/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tubelet_embedding_31/conv3d_95/kernel/v*+
_output_shapes
:@�*
dtype0
�
*Adam/tubelet_embedding_31/conv3d_95/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*Adam/tubelet_embedding_31/conv3d_95/bias/v
�
>Adam/tubelet_embedding_31/conv3d_95/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tubelet_embedding_31/conv3d_95/bias/v*
_output_shapes	
:�*
dtype0
�
1Adam/positional_encoder_15/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(�*B
shared_name31Adam/positional_encoder_15/embedding/embeddings/v
�
EAdam/positional_encoder_15/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp1Adam/positional_encoder_15/embedding/embeddings/v*
_output_shapes
:	(�*
dtype0
�
*Adam/multi_head_attention_5/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*;
shared_name,*Adam/multi_head_attention_5/query/kernel/v
�
>Adam/multi_head_attention_5/query/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_5/query/kernel/v*$
_output_shapes
:��*
dtype0
�
(Adam/multi_head_attention_5/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*9
shared_name*(Adam/multi_head_attention_5/query/bias/v
�
<Adam/multi_head_attention_5/query/bias/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_5/query/bias/v*
_output_shapes
:	�*
dtype0
�
(Adam/multi_head_attention_5/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*9
shared_name*(Adam/multi_head_attention_5/key/kernel/v
�
<Adam/multi_head_attention_5/key/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_5/key/kernel/v*$
_output_shapes
:��*
dtype0
�
&Adam/multi_head_attention_5/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*7
shared_name(&Adam/multi_head_attention_5/key/bias/v
�
:Adam/multi_head_attention_5/key/bias/v/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention_5/key/bias/v*
_output_shapes
:	�*
dtype0
�
*Adam/multi_head_attention_5/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*;
shared_name,*Adam/multi_head_attention_5/value/kernel/v
�
>Adam/multi_head_attention_5/value/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_5/value/kernel/v*$
_output_shapes
:��*
dtype0
�
(Adam/multi_head_attention_5/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*9
shared_name*(Adam/multi_head_attention_5/value/bias/v
�
<Adam/multi_head_attention_5/value/bias/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_5/value/bias/v*
_output_shapes
:	�*
dtype0
�
5Adam/multi_head_attention_5/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*F
shared_name75Adam/multi_head_attention_5/attention_output/kernel/v
�
IAdam/multi_head_attention_5/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/multi_head_attention_5/attention_output/kernel/v*$
_output_shapes
:��*
dtype0
�
3Adam/multi_head_attention_5/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53Adam/multi_head_attention_5/attention_output/bias/v
�
GAdam/multi_head_attention_5/attention_output/bias/v/Read/ReadVariableOpReadVariableOp3Adam/multi_head_attention_5/attention_output/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_10/kernel/v
�
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_10/bias/v
z
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes	
:�*
dtype0
�
ConstConst*
_output_shapes
:(*
dtype0*�
value�B�("�                            	   
                                                                      !   "   #   $   %   &   '   

NoOpNoOp
��
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
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
�
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
�
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
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
�
<position_embedding
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
�
Caxis
	Dgamma
Ebeta
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*
�
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
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 
�
^axis
	_gamma
`beta
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses*
�
glayer_with_weights-0
glayer-0
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses*
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 
�
taxis
	ugamma
vbeta
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses*
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
	�iter
�beta_1
�beta_2

�decay
�learning_rateDm�Em�_m�`m�um�vm�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�Dv�Ev�_v�`v�uv�vv�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
D13
E14
�15
�16
�17
�18
�19
�20
�21
�22
_23
`24
�25
�26
u27
v28
�29
�30*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
D13
E14
�15
�16
�17
�18
�19
�20
�21
�22
_23
`24
�25
�26
u27
v28
�29
�30*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
�serving_default* 
* 
* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
4
�0
�1
�2
�3
�4
�5*
4
�0
�1
�2
�3
�4
�5*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
* 
* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
4
�0
�1
�2
�3
�4
�5*
4
�0
�1
�2
�3
�4
�5*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
�
�
embeddings
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
VARIABLE_VALUElayer_normalization_15/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_15/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses* 
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
VARIABLE_VALUElayer_normalization_16/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_16/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*

_0
`1*

_0
`1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
VARIABLE_VALUElayer_normalization_17/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_17/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

u0
v1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
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
VARIABLE_VALUE%tubelet_embedding_30/conv3d_90/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#tubelet_embedding_30/conv3d_90/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%tubelet_embedding_30/conv3d_91/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#tubelet_embedding_30/conv3d_91/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%tubelet_embedding_30/conv3d_92/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#tubelet_embedding_30/conv3d_92/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%tubelet_embedding_31/conv3d_93/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#tubelet_embedding_31/conv3d_93/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%tubelet_embedding_31/conv3d_94/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#tubelet_embedding_31/conv3d_94/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%tubelet_embedding_31/conv3d_95/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#tubelet_embedding_31/conv3d_95/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*positional_encoder_15/embedding/embeddings'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_5/query/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_5/query/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_5/key/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_5/key/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_5/value/kernel'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_5/value/bias'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_5/attention_output/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_5/attention_output/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_10/kernel'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_10/bias'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
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
�0
�1*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
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
�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
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

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
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
�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
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
�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
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

�total

�count
�	variables
�	keras_api*
M

�total

�count
�
_fn_kwargs
�	variables
�	keras_api*
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
�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
��
VARIABLE_VALUE#Adam/layer_normalization_15/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/layer_normalization_15/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/layer_normalization_16/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/layer_normalization_16/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/layer_normalization_17/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/layer_normalization_17/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_30/conv3d_90/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_30/conv3d_90/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_30/conv3d_91/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_30/conv3d_91/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_30/conv3d_92/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_30/conv3d_92/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_31/conv3d_93/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_31/conv3d_93/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_31/conv3d_94/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_31/conv3d_94/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_31/conv3d_95/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_31/conv3d_95/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE1Adam/positional_encoder_15/embedding/embeddings/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/multi_head_attention_5/query/kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE(Adam/multi_head_attention_5/query/bias/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE(Adam/multi_head_attention_5/key/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUE&Adam/multi_head_attention_5/key/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/multi_head_attention_5/value/kernel/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE(Adam/multi_head_attention_5/value/bias/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/multi_head_attention_5/attention_output/kernel/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/multi_head_attention_5/attention_output/bias/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_10/kernel/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_10/bias/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/layer_normalization_15/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/layer_normalization_15/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/layer_normalization_16/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/layer_normalization_16/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/layer_normalization_17/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/layer_normalization_17/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_30/conv3d_90/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_30/conv3d_90/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_30/conv3d_91/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_30/conv3d_91/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_30/conv3d_92/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_30/conv3d_92/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_31/conv3d_93/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_31/conv3d_93/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_31/conv3d_94/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_31/conv3d_94/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/tubelet_embedding_31/conv3d_95/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/tubelet_embedding_31/conv3d_95/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE1Adam/positional_encoder_15/embedding/embeddings/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/multi_head_attention_5/query/kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE(Adam/multi_head_attention_5/query/bias/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE(Adam/multi_head_attention_5/key/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUE&Adam/multi_head_attention_5/key/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/multi_head_attention_5/value/kernel/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE(Adam/multi_head_attention_5/value/bias/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/multi_head_attention_5/attention_output/kernel/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/multi_head_attention_5/attention_output/bias/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_10/kernel/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_10/bias/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_input_16Placeholder*3
_output_shapes!
:���������
*
dtype0*(
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_16%tubelet_embedding_30/conv3d_90/kernel#tubelet_embedding_30/conv3d_90/bias%tubelet_embedding_30/conv3d_91/kernel#tubelet_embedding_30/conv3d_91/bias%tubelet_embedding_30/conv3d_92/kernel#tubelet_embedding_30/conv3d_92/bias%tubelet_embedding_31/conv3d_93/kernel#tubelet_embedding_31/conv3d_93/bias%tubelet_embedding_31/conv3d_94/kernel#tubelet_embedding_31/conv3d_94/bias%tubelet_embedding_31/conv3d_95/kernel#tubelet_embedding_31/conv3d_95/biasConst*positional_encoder_15/embedding/embeddingslayer_normalization_15/gammalayer_normalization_15/beta#multi_head_attention_5/query/kernel!multi_head_attention_5/query/bias!multi_head_attention_5/key/kernelmulti_head_attention_5/key/bias#multi_head_attention_5/value/kernel!multi_head_attention_5/value/bias.multi_head_attention_5/attention_output/kernel,multi_head_attention_5/attention_output/biaslayer_normalization_16/gammalayer_normalization_16/betadense_10/kerneldense_10/biaslayer_normalization_17/gammalayer_normalization_17/betadense_11/kerneldense_11/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8� *-
f(R&
$__inference_signature_wrapper_384693
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�/
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0layer_normalization_15/gamma/Read/ReadVariableOp/layer_normalization_15/beta/Read/ReadVariableOp0layer_normalization_16/gamma/Read/ReadVariableOp/layer_normalization_16/beta/Read/ReadVariableOp0layer_normalization_17/gamma/Read/ReadVariableOp/layer_normalization_17/beta/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9tubelet_embedding_30/conv3d_90/kernel/Read/ReadVariableOp7tubelet_embedding_30/conv3d_90/bias/Read/ReadVariableOp9tubelet_embedding_30/conv3d_91/kernel/Read/ReadVariableOp7tubelet_embedding_30/conv3d_91/bias/Read/ReadVariableOp9tubelet_embedding_30/conv3d_92/kernel/Read/ReadVariableOp7tubelet_embedding_30/conv3d_92/bias/Read/ReadVariableOp9tubelet_embedding_31/conv3d_93/kernel/Read/ReadVariableOp7tubelet_embedding_31/conv3d_93/bias/Read/ReadVariableOp9tubelet_embedding_31/conv3d_94/kernel/Read/ReadVariableOp7tubelet_embedding_31/conv3d_94/bias/Read/ReadVariableOp9tubelet_embedding_31/conv3d_95/kernel/Read/ReadVariableOp7tubelet_embedding_31/conv3d_95/bias/Read/ReadVariableOp>positional_encoder_15/embedding/embeddings/Read/ReadVariableOp7multi_head_attention_5/query/kernel/Read/ReadVariableOp5multi_head_attention_5/query/bias/Read/ReadVariableOp5multi_head_attention_5/key/kernel/Read/ReadVariableOp3multi_head_attention_5/key/bias/Read/ReadVariableOp7multi_head_attention_5/value/kernel/Read/ReadVariableOp5multi_head_attention_5/value/bias/Read/ReadVariableOpBmulti_head_attention_5/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_5/attention_output/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adam/layer_normalization_15/gamma/m/Read/ReadVariableOp6Adam/layer_normalization_15/beta/m/Read/ReadVariableOp7Adam/layer_normalization_16/gamma/m/Read/ReadVariableOp6Adam/layer_normalization_16/beta/m/Read/ReadVariableOp7Adam/layer_normalization_17/gamma/m/Read/ReadVariableOp6Adam/layer_normalization_17/beta/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp@Adam/tubelet_embedding_30/conv3d_90/kernel/m/Read/ReadVariableOp>Adam/tubelet_embedding_30/conv3d_90/bias/m/Read/ReadVariableOp@Adam/tubelet_embedding_30/conv3d_91/kernel/m/Read/ReadVariableOp>Adam/tubelet_embedding_30/conv3d_91/bias/m/Read/ReadVariableOp@Adam/tubelet_embedding_30/conv3d_92/kernel/m/Read/ReadVariableOp>Adam/tubelet_embedding_30/conv3d_92/bias/m/Read/ReadVariableOp@Adam/tubelet_embedding_31/conv3d_93/kernel/m/Read/ReadVariableOp>Adam/tubelet_embedding_31/conv3d_93/bias/m/Read/ReadVariableOp@Adam/tubelet_embedding_31/conv3d_94/kernel/m/Read/ReadVariableOp>Adam/tubelet_embedding_31/conv3d_94/bias/m/Read/ReadVariableOp@Adam/tubelet_embedding_31/conv3d_95/kernel/m/Read/ReadVariableOp>Adam/tubelet_embedding_31/conv3d_95/bias/m/Read/ReadVariableOpEAdam/positional_encoder_15/embedding/embeddings/m/Read/ReadVariableOp>Adam/multi_head_attention_5/query/kernel/m/Read/ReadVariableOp<Adam/multi_head_attention_5/query/bias/m/Read/ReadVariableOp<Adam/multi_head_attention_5/key/kernel/m/Read/ReadVariableOp:Adam/multi_head_attention_5/key/bias/m/Read/ReadVariableOp>Adam/multi_head_attention_5/value/kernel/m/Read/ReadVariableOp<Adam/multi_head_attention_5/value/bias/m/Read/ReadVariableOpIAdam/multi_head_attention_5/attention_output/kernel/m/Read/ReadVariableOpGAdam/multi_head_attention_5/attention_output/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp7Adam/layer_normalization_15/gamma/v/Read/ReadVariableOp6Adam/layer_normalization_15/beta/v/Read/ReadVariableOp7Adam/layer_normalization_16/gamma/v/Read/ReadVariableOp6Adam/layer_normalization_16/beta/v/Read/ReadVariableOp7Adam/layer_normalization_17/gamma/v/Read/ReadVariableOp6Adam/layer_normalization_17/beta/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOp@Adam/tubelet_embedding_30/conv3d_90/kernel/v/Read/ReadVariableOp>Adam/tubelet_embedding_30/conv3d_90/bias/v/Read/ReadVariableOp@Adam/tubelet_embedding_30/conv3d_91/kernel/v/Read/ReadVariableOp>Adam/tubelet_embedding_30/conv3d_91/bias/v/Read/ReadVariableOp@Adam/tubelet_embedding_30/conv3d_92/kernel/v/Read/ReadVariableOp>Adam/tubelet_embedding_30/conv3d_92/bias/v/Read/ReadVariableOp@Adam/tubelet_embedding_31/conv3d_93/kernel/v/Read/ReadVariableOp>Adam/tubelet_embedding_31/conv3d_93/bias/v/Read/ReadVariableOp@Adam/tubelet_embedding_31/conv3d_94/kernel/v/Read/ReadVariableOp>Adam/tubelet_embedding_31/conv3d_94/bias/v/Read/ReadVariableOp@Adam/tubelet_embedding_31/conv3d_95/kernel/v/Read/ReadVariableOp>Adam/tubelet_embedding_31/conv3d_95/bias/v/Read/ReadVariableOpEAdam/positional_encoder_15/embedding/embeddings/v/Read/ReadVariableOp>Adam/multi_head_attention_5/query/kernel/v/Read/ReadVariableOp<Adam/multi_head_attention_5/query/bias/v/Read/ReadVariableOp<Adam/multi_head_attention_5/key/kernel/v/Read/ReadVariableOp:Adam/multi_head_attention_5/key/bias/v/Read/ReadVariableOp>Adam/multi_head_attention_5/value/kernel/v/Read/ReadVariableOp<Adam/multi_head_attention_5/value/bias/v/Read/ReadVariableOpIAdam/multi_head_attention_5/attention_output/kernel/v/Read/ReadVariableOpGAdam/multi_head_attention_5/attention_output/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOpConst_1*s
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
GPU2*0,1J 8� *(
f#R!
__inference__traced_save_385638
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization_15/gammalayer_normalization_15/betalayer_normalization_16/gammalayer_normalization_16/betalayer_normalization_17/gammalayer_normalization_17/betadense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%tubelet_embedding_30/conv3d_90/kernel#tubelet_embedding_30/conv3d_90/bias%tubelet_embedding_30/conv3d_91/kernel#tubelet_embedding_30/conv3d_91/bias%tubelet_embedding_30/conv3d_92/kernel#tubelet_embedding_30/conv3d_92/bias%tubelet_embedding_31/conv3d_93/kernel#tubelet_embedding_31/conv3d_93/bias%tubelet_embedding_31/conv3d_94/kernel#tubelet_embedding_31/conv3d_94/bias%tubelet_embedding_31/conv3d_95/kernel#tubelet_embedding_31/conv3d_95/bias*positional_encoder_15/embedding/embeddings#multi_head_attention_5/query/kernel!multi_head_attention_5/query/bias!multi_head_attention_5/key/kernelmulti_head_attention_5/key/bias#multi_head_attention_5/value/kernel!multi_head_attention_5/value/bias.multi_head_attention_5/attention_output/kernel,multi_head_attention_5/attention_output/biasdense_10/kerneldense_10/biastotalcounttotal_1count_1#Adam/layer_normalization_15/gamma/m"Adam/layer_normalization_15/beta/m#Adam/layer_normalization_16/gamma/m"Adam/layer_normalization_16/beta/m#Adam/layer_normalization_17/gamma/m"Adam/layer_normalization_17/beta/mAdam/dense_11/kernel/mAdam/dense_11/bias/m,Adam/tubelet_embedding_30/conv3d_90/kernel/m*Adam/tubelet_embedding_30/conv3d_90/bias/m,Adam/tubelet_embedding_30/conv3d_91/kernel/m*Adam/tubelet_embedding_30/conv3d_91/bias/m,Adam/tubelet_embedding_30/conv3d_92/kernel/m*Adam/tubelet_embedding_30/conv3d_92/bias/m,Adam/tubelet_embedding_31/conv3d_93/kernel/m*Adam/tubelet_embedding_31/conv3d_93/bias/m,Adam/tubelet_embedding_31/conv3d_94/kernel/m*Adam/tubelet_embedding_31/conv3d_94/bias/m,Adam/tubelet_embedding_31/conv3d_95/kernel/m*Adam/tubelet_embedding_31/conv3d_95/bias/m1Adam/positional_encoder_15/embedding/embeddings/m*Adam/multi_head_attention_5/query/kernel/m(Adam/multi_head_attention_5/query/bias/m(Adam/multi_head_attention_5/key/kernel/m&Adam/multi_head_attention_5/key/bias/m*Adam/multi_head_attention_5/value/kernel/m(Adam/multi_head_attention_5/value/bias/m5Adam/multi_head_attention_5/attention_output/kernel/m3Adam/multi_head_attention_5/attention_output/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/m#Adam/layer_normalization_15/gamma/v"Adam/layer_normalization_15/beta/v#Adam/layer_normalization_16/gamma/v"Adam/layer_normalization_16/beta/v#Adam/layer_normalization_17/gamma/v"Adam/layer_normalization_17/beta/vAdam/dense_11/kernel/vAdam/dense_11/bias/v,Adam/tubelet_embedding_30/conv3d_90/kernel/v*Adam/tubelet_embedding_30/conv3d_90/bias/v,Adam/tubelet_embedding_30/conv3d_91/kernel/v*Adam/tubelet_embedding_30/conv3d_91/bias/v,Adam/tubelet_embedding_30/conv3d_92/kernel/v*Adam/tubelet_embedding_30/conv3d_92/bias/v,Adam/tubelet_embedding_31/conv3d_93/kernel/v*Adam/tubelet_embedding_31/conv3d_93/bias/v,Adam/tubelet_embedding_31/conv3d_94/kernel/v*Adam/tubelet_embedding_31/conv3d_94/bias/v,Adam/tubelet_embedding_31/conv3d_95/kernel/v*Adam/tubelet_embedding_31/conv3d_95/bias/v1Adam/positional_encoder_15/embedding/embeddings/v*Adam/multi_head_attention_5/query/kernel/v(Adam/multi_head_attention_5/query/bias/v(Adam/multi_head_attention_5/key/kernel/v&Adam/multi_head_attention_5/key/bias/v*Adam/multi_head_attention_5/value/kernel/v(Adam/multi_head_attention_5/value/bias/v5Adam/multi_head_attention_5/attention_output/kernel/v3Adam/multi_head_attention_5/attention_output/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/v*r
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
GPU2*0,1J 8� *+
f&R$
"__inference__traced_restore_385954��
�
h
L__inference_max_pooling3d_61_layer_call_and_return_conditional_losses_382793

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
-__inference_sequential_5_layer_call_fn_382949
dense_10_input
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_10_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_382933t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:���������(�
(
_user_specified_namedense_10_input
�
h
L__inference_max_pooling3d_60_layer_call_and_return_conditional_losses_382781

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
v
J__inference_concatenate_12_layer_call_and_return_conditional_losses_384812
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
:���������(�\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:���������(�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������(�:���������(�:V R
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/1
�
n
B__inference_add_11_layer_call_and_return_conditional_losses_385140
inputs_0
inputs_1
identityW
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:���������(�T
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:���������(�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������(�:���������(�:V R
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/1
�(
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_385128

inputs>
*dense_10_tensordot_readvariableop_resource:
��7
(dense_10_biasadd_readvariableop_resource:	�
identity��dense_10/BiasAdd/ReadVariableOp�!dense_10/Tensordot/ReadVariableOp�
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0a
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_10/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_10/Tensordot/transpose	Transposeinputs"dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������(��
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�b
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������(��
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�X
dense_10/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dense_10/Gelu/mulMuldense_10/Gelu/mul/x:output:0dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:���������(�Y
dense_10/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dense_10/Gelu/truedivRealDivdense_10/BiasAdd:output:0dense_10/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������(�j
dense_10/Gelu/ErfErfdense_10/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������(�X
dense_10/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dense_10/Gelu/addAddV2dense_10/Gelu/add/x:output:0dense_10/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������(�
dense_10/Gelu/mul_1Muldense_10/Gelu/mul:z:0dense_10/Gelu/add:z:0*
T0*,
_output_shapes
:���������(�k
IdentityIdentitydense_10/Gelu/mul_1:z:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
M
1__inference_max_pooling3d_61_layer_call_fn_385216

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_max_pooling3d_61_layer_call_and_return_conditional_losses_382793�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling3d_62_layer_call_and_return_conditional_losses_385241

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
(__inference_model_5_layer_call_fn_384187

inputs%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�'
	unknown_5: 
	unknown_6: '
	unknown_7: @
	unknown_8:@(
	unknown_9:@�

unknown_10:	�

unknown_11

unknown_12:	(�

unknown_13:	�

unknown_14:	�"

unknown_15:��

unknown_16:	�"

unknown_17:��

unknown_18:	�"

unknown_19:��

unknown_20:	�"

unknown_21:��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�

unknown_30:
identity��StatefulPartitionedCall�
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
:���������*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_383727o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������

 
_user_specified_nameinputs: 

_output_shapes
:(
�
�
R__inference_layer_normalization_16_layer_call_and_return_conditional_losses_385034

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������(�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������(�l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�h
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(�g
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
l
P__inference_average_pooling3d_31_layer_call_and_return_conditional_losses_385261

inputs
identity�
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
-__inference_sequential_5_layer_call_fn_385052

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_382933t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�[
�
C__inference_model_5_layer_call_and_return_conditional_losses_383953
input_169
tubelet_embedding_30_383874: )
tubelet_embedding_30_383876: 9
tubelet_embedding_30_383878: @)
tubelet_embedding_30_383880:@:
tubelet_embedding_30_383882:@�*
tubelet_embedding_30_383884:	�9
tubelet_embedding_31_383887: )
tubelet_embedding_31_383889: 9
tubelet_embedding_31_383891: @)
tubelet_embedding_31_383893:@:
tubelet_embedding_31_383895:@�*
tubelet_embedding_31_383897:	� 
positional_encoder_15_383901/
positional_encoder_15_383903:	(�,
layer_normalization_15_383906:	�,
layer_normalization_15_383908:	�5
multi_head_attention_5_383911:��0
multi_head_attention_5_383913:	�5
multi_head_attention_5_383915:��0
multi_head_attention_5_383917:	�5
multi_head_attention_5_383919:��0
multi_head_attention_5_383921:	�5
multi_head_attention_5_383923:��,
multi_head_attention_5_383925:	�,
layer_normalization_16_383930:	�,
layer_normalization_16_383932:	�'
sequential_5_383935:
��"
sequential_5_383937:	�,
layer_normalization_17_383941:	�,
layer_normalization_17_383943:	�"
dense_11_383947:	�
dense_11_383949:
identity�� dense_11/StatefulPartitionedCall�.layer_normalization_15/StatefulPartitionedCall�.layer_normalization_16/StatefulPartitionedCall�.layer_normalization_17/StatefulPartitionedCall�.multi_head_attention_5/StatefulPartitionedCall�-positional_encoder_15/StatefulPartitionedCall�$sequential_5/StatefulPartitionedCall�,tubelet_embedding_30/StatefulPartitionedCall�,tubelet_embedding_31/StatefulPartitionedCall�
/tf.__operators__.getitem_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_35/strided_sliceStridedSliceinput_168tf.__operators__.getitem_35/strided_slice/stack:output:0:tf.__operators__.getitem_35/strided_slice/stack_1:output:0:tf.__operators__.getitem_35/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
/tf.__operators__.getitem_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_34/strided_sliceStridedSliceinput_168tf.__operators__.getitem_34/strided_slice/stack:output:0:tf.__operators__.getitem_34/strided_slice/stack_1:output:0:tf.__operators__.getitem_34/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
,tubelet_embedding_30/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_34/strided_slice:output:0tubelet_embedding_30_383874tubelet_embedding_30_383876tubelet_embedding_30_383878tubelet_embedding_30_383880tubelet_embedding_30_383882tubelet_embedding_30_383884*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_tubelet_embedding_30_layer_call_and_return_conditional_losses_383031�
,tubelet_embedding_31/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_35/strided_slice:output:0tubelet_embedding_31_383887tubelet_embedding_31_383889tubelet_embedding_31_383891tubelet_embedding_31_383893tubelet_embedding_31_383895tubelet_embedding_31_383897*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_tubelet_embedding_31_layer_call_and_return_conditional_losses_383081�
concatenate_12/PartitionedCallPartitionedCall5tubelet_embedding_30/StatefulPartitionedCall:output:05tubelet_embedding_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *S
fNRL
J__inference_concatenate_12_layer_call_and_return_conditional_losses_383102�
-positional_encoder_15/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0positional_encoder_15_383901positional_encoder_15_383903*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Z
fURS
Q__inference_positional_encoder_15_layer_call_and_return_conditional_losses_383116�
.layer_normalization_15/StatefulPartitionedCallStatefulPartitionedCall6positional_encoder_15/StatefulPartitionedCall:output:0layer_normalization_15_383906layer_normalization_15_383908*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_383144�
.multi_head_attention_5/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_15/StatefulPartitionedCall:output:07layer_normalization_15/StatefulPartitionedCall:output:0multi_head_attention_5_383911multi_head_attention_5_383913multi_head_attention_5_383915multi_head_attention_5_383917multi_head_attention_5_383919multi_head_attention_5_383921multi_head_attention_5_383923multi_head_attention_5_383925*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:���������(�:���������((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_383186�
add_10/PartitionedCallPartitionedCall7multi_head_attention_5/StatefulPartitionedCall:output:06positional_encoder_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_add_10_layer_call_and_return_conditional_losses_383211�
.layer_normalization_16/StatefulPartitionedCallStatefulPartitionedCalladd_10/PartitionedCall:output:0layer_normalization_16_383930layer_normalization_16_383932*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_16_layer_call_and_return_conditional_losses_383235�
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_16/StatefulPartitionedCall:output:0sequential_5_383935sequential_5_383937*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_382896�
add_11/PartitionedCallPartitionedCall-sequential_5/StatefulPartitionedCall:output:0add_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_add_11_layer_call_and_return_conditional_losses_383252�
.layer_normalization_17/StatefulPartitionedCallStatefulPartitionedCalladd_11/PartitionedCall:output:0layer_normalization_17_383941layer_normalization_17_383943*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_17_layer_call_and_return_conditional_losses_383276�
*global_average_pooling1d_5/PartitionedCallPartitionedCall7layer_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *_
fZRX
V__inference_global_average_pooling1d_5_layer_call_and_return_conditional_losses_382977�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_5/PartitionedCall:output:0dense_11_383947dense_11_383949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_383293x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_11/StatefulPartitionedCall/^layer_normalization_15/StatefulPartitionedCall/^layer_normalization_16/StatefulPartitionedCall/^layer_normalization_17/StatefulPartitionedCall/^multi_head_attention_5/StatefulPartitionedCall.^positional_encoder_15/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall-^tubelet_embedding_30/StatefulPartitionedCall-^tubelet_embedding_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2`
.layer_normalization_15/StatefulPartitionedCall.layer_normalization_15/StatefulPartitionedCall2`
.layer_normalization_16/StatefulPartitionedCall.layer_normalization_16/StatefulPartitionedCall2`
.layer_normalization_17/StatefulPartitionedCall.layer_normalization_17/StatefulPartitionedCall2`
.multi_head_attention_5/StatefulPartitionedCall.multi_head_attention_5/StatefulPartitionedCall2^
-positional_encoder_15/StatefulPartitionedCall-positional_encoder_15/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2\
,tubelet_embedding_30/StatefulPartitionedCall,tubelet_embedding_30/StatefulPartitionedCall2\
,tubelet_embedding_31/StatefulPartitionedCall,tubelet_embedding_31/StatefulPartitionedCall:] Y
3
_output_shapes!
:���������

"
_user_specified_name
input_16: 

_output_shapes
:(
�
�
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_383144

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������(�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������(�l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�h
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(�g
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_382967
dense_10_input#
dense_10_382961:
��
dense_10_382963:	�
identity�� dense_10/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCalldense_10_inputdense_10_382961dense_10_382963*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_382889}
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�i
NoOpNoOp!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:\ X
,
_output_shapes
:���������(�
(
_user_specified_namedense_10_input
�
�
$__inference_signature_wrapper_384693
input_16%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�'
	unknown_5: 
	unknown_6: '
	unknown_7: @
	unknown_8:@(
	unknown_9:@�

unknown_10:	�

unknown_11

unknown_12:	(�

unknown_13:	�

unknown_14:	�"

unknown_15:��

unknown_16:	�"

unknown_17:��

unknown_18:	�"

unknown_19:��

unknown_20:	�"

unknown_21:��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:���������*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8� **
f%R#
!__inference__wrapped_model_382772o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
3
_output_shapes!
:���������

"
_user_specified_name
input_16: 

_output_shapes
:(
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_382896

inputs#
dense_10_382890:
��
dense_10_382892:	�
identity�� dense_10/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_382890dense_10_382892*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_382889}
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�i
NoOpNoOp!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�	
�
5__inference_tubelet_embedding_31_layer_call_fn_384763

videos%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvideosunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_tubelet_embedding_31_layer_call_and_return_conditional_losses_383081t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������

 
_user_specified_namevideos
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_382958
dense_10_input#
dense_10_382952:
��
dense_10_382954:	�
identity�� dense_10/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCalldense_10_inputdense_10_382952dense_10_382954*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_382889}
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�i
NoOpNoOp!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:\ X
,
_output_shapes
:���������(�
(
_user_specified_namedense_10_input
�"
�
D__inference_dense_10_layer_call_and_return_conditional_losses_382889

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
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
value	B : �
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
value	B : �
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
value	B : �
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
:���������(��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������(�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*,
_output_shapes
:���������(�P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?v
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������(�X
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:���������(�O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:���������(�d

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:���������(�b
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*,
_output_shapes
:���������(�z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
�
)__inference_dense_11_layer_call_fn_385191

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_383293o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling3d_60_layer_call_fn_385206

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_max_pooling3d_60_layer_call_and_return_conditional_losses_382781�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�[
�
C__inference_model_5_layer_call_and_return_conditional_losses_383300

inputs9
tubelet_embedding_30_383032: )
tubelet_embedding_30_383034: 9
tubelet_embedding_30_383036: @)
tubelet_embedding_30_383038:@:
tubelet_embedding_30_383040:@�*
tubelet_embedding_30_383042:	�9
tubelet_embedding_31_383082: )
tubelet_embedding_31_383084: 9
tubelet_embedding_31_383086: @)
tubelet_embedding_31_383088:@:
tubelet_embedding_31_383090:@�*
tubelet_embedding_31_383092:	� 
positional_encoder_15_383117/
positional_encoder_15_383119:	(�,
layer_normalization_15_383145:	�,
layer_normalization_15_383147:	�5
multi_head_attention_5_383187:��0
multi_head_attention_5_383189:	�5
multi_head_attention_5_383191:��0
multi_head_attention_5_383193:	�5
multi_head_attention_5_383195:��0
multi_head_attention_5_383197:	�5
multi_head_attention_5_383199:��,
multi_head_attention_5_383201:	�,
layer_normalization_16_383236:	�,
layer_normalization_16_383238:	�'
sequential_5_383241:
��"
sequential_5_383243:	�,
layer_normalization_17_383277:	�,
layer_normalization_17_383279:	�"
dense_11_383294:	�
dense_11_383296:
identity�� dense_11/StatefulPartitionedCall�.layer_normalization_15/StatefulPartitionedCall�.layer_normalization_16/StatefulPartitionedCall�.layer_normalization_17/StatefulPartitionedCall�.multi_head_attention_5/StatefulPartitionedCall�-positional_encoder_15/StatefulPartitionedCall�$sequential_5/StatefulPartitionedCall�,tubelet_embedding_30/StatefulPartitionedCall�,tubelet_embedding_31/StatefulPartitionedCall�
/tf.__operators__.getitem_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_35/strided_sliceStridedSliceinputs8tf.__operators__.getitem_35/strided_slice/stack:output:0:tf.__operators__.getitem_35/strided_slice/stack_1:output:0:tf.__operators__.getitem_35/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
/tf.__operators__.getitem_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_34/strided_sliceStridedSliceinputs8tf.__operators__.getitem_34/strided_slice/stack:output:0:tf.__operators__.getitem_34/strided_slice/stack_1:output:0:tf.__operators__.getitem_34/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
,tubelet_embedding_30/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_34/strided_slice:output:0tubelet_embedding_30_383032tubelet_embedding_30_383034tubelet_embedding_30_383036tubelet_embedding_30_383038tubelet_embedding_30_383040tubelet_embedding_30_383042*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_tubelet_embedding_30_layer_call_and_return_conditional_losses_383031�
,tubelet_embedding_31/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_35/strided_slice:output:0tubelet_embedding_31_383082tubelet_embedding_31_383084tubelet_embedding_31_383086tubelet_embedding_31_383088tubelet_embedding_31_383090tubelet_embedding_31_383092*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_tubelet_embedding_31_layer_call_and_return_conditional_losses_383081�
concatenate_12/PartitionedCallPartitionedCall5tubelet_embedding_30/StatefulPartitionedCall:output:05tubelet_embedding_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *S
fNRL
J__inference_concatenate_12_layer_call_and_return_conditional_losses_383102�
-positional_encoder_15/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0positional_encoder_15_383117positional_encoder_15_383119*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Z
fURS
Q__inference_positional_encoder_15_layer_call_and_return_conditional_losses_383116�
.layer_normalization_15/StatefulPartitionedCallStatefulPartitionedCall6positional_encoder_15/StatefulPartitionedCall:output:0layer_normalization_15_383145layer_normalization_15_383147*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_383144�
.multi_head_attention_5/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_15/StatefulPartitionedCall:output:07layer_normalization_15/StatefulPartitionedCall:output:0multi_head_attention_5_383187multi_head_attention_5_383189multi_head_attention_5_383191multi_head_attention_5_383193multi_head_attention_5_383195multi_head_attention_5_383197multi_head_attention_5_383199multi_head_attention_5_383201*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:���������(�:���������((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_383186�
add_10/PartitionedCallPartitionedCall7multi_head_attention_5/StatefulPartitionedCall:output:06positional_encoder_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_add_10_layer_call_and_return_conditional_losses_383211�
.layer_normalization_16/StatefulPartitionedCallStatefulPartitionedCalladd_10/PartitionedCall:output:0layer_normalization_16_383236layer_normalization_16_383238*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_16_layer_call_and_return_conditional_losses_383235�
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_16/StatefulPartitionedCall:output:0sequential_5_383241sequential_5_383243*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_382896�
add_11/PartitionedCallPartitionedCall-sequential_5/StatefulPartitionedCall:output:0add_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_add_11_layer_call_and_return_conditional_losses_383252�
.layer_normalization_17/StatefulPartitionedCallStatefulPartitionedCalladd_11/PartitionedCall:output:0layer_normalization_17_383277layer_normalization_17_383279*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_17_layer_call_and_return_conditional_losses_383276�
*global_average_pooling1d_5/PartitionedCallPartitionedCall7layer_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *_
fZRX
V__inference_global_average_pooling1d_5_layer_call_and_return_conditional_losses_382977�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_5/PartitionedCall:output:0dense_11_383294dense_11_383296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_383293x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_11/StatefulPartitionedCall/^layer_normalization_15/StatefulPartitionedCall/^layer_normalization_16/StatefulPartitionedCall/^layer_normalization_17/StatefulPartitionedCall/^multi_head_attention_5/StatefulPartitionedCall.^positional_encoder_15/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall-^tubelet_embedding_30/StatefulPartitionedCall-^tubelet_embedding_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2`
.layer_normalization_15/StatefulPartitionedCall.layer_normalization_15/StatefulPartitionedCall2`
.layer_normalization_16/StatefulPartitionedCall.layer_normalization_16/StatefulPartitionedCall2`
.layer_normalization_17/StatefulPartitionedCall.layer_normalization_17/StatefulPartitionedCall2`
.multi_head_attention_5/StatefulPartitionedCall.multi_head_attention_5/StatefulPartitionedCall2^
-positional_encoder_15/StatefulPartitionedCall-positional_encoder_15/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2\
,tubelet_embedding_30/StatefulPartitionedCall,tubelet_embedding_30/StatefulPartitionedCall2\
,tubelet_embedding_31/StatefulPartitionedCall,tubelet_embedding_31/StatefulPartitionedCall:[ W
3
_output_shapes!
:���������

 
_user_specified_nameinputs: 

_output_shapes
:(
�-
�
P__inference_tubelet_embedding_31_layer_call_and_return_conditional_losses_384799

videosF
(conv3d_93_conv3d_readvariableop_resource: 7
)conv3d_93_biasadd_readvariableop_resource: F
(conv3d_94_conv3d_readvariableop_resource: @7
)conv3d_94_biasadd_readvariableop_resource:@G
(conv3d_95_conv3d_readvariableop_resource:@�8
)conv3d_95_biasadd_readvariableop_resource:	�
identity�� conv3d_93/BiasAdd/ReadVariableOp�conv3d_93/Conv3D/ReadVariableOp� conv3d_94/BiasAdd/ReadVariableOp�conv3d_94/Conv3D/ReadVariableOp� conv3d_95/BiasAdd/ReadVariableOp�conv3d_95/Conv3D/ReadVariableOp�
conv3d_93/Conv3D/ReadVariableOpReadVariableOp(conv3d_93_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
conv3d_93/Conv3DConv3Dvideos'conv3d_93/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
 conv3d_93/BiasAdd/ReadVariableOpReadVariableOp)conv3d_93_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv3d_93/BiasAddBiasAddconv3d_93/Conv3D:output:0(conv3d_93/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 p
conv3d_93/ReluReluconv3d_93/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
max_pooling3d_62/MaxPool3D	MaxPool3Dconv3d_93/Relu:activations:0*
T0*3
_output_shapes!
:���������
 *
ksize	
*
paddingVALID*
strides	
�
conv3d_94/Conv3D/ReadVariableOpReadVariableOp(conv3d_94_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
conv3d_94/Conv3DConv3D#max_pooling3d_62/MaxPool3D:output:0'conv3d_94/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@*
paddingSAME*
strides	
�
 conv3d_94/BiasAdd/ReadVariableOpReadVariableOp)conv3d_94_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d_94/BiasAddBiasAddconv3d_94/Conv3D:output:0(conv3d_94/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@p
conv3d_94/ReluReluconv3d_94/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
@�
max_pooling3d_63/MaxPool3D	MaxPool3Dconv3d_94/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
conv3d_95/Conv3D/ReadVariableOpReadVariableOp(conv3d_95_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
conv3d_95/Conv3DConv3D#max_pooling3d_63/MaxPool3D:output:0'conv3d_95/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
 conv3d_95/BiasAdd/ReadVariableOpReadVariableOp)conv3d_95_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_95/BiasAddBiasAddconv3d_95/Conv3D:output:0(conv3d_95/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
average_pooling3d_31/AvgPool3D	AvgPool3Dconv3d_95/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
�*
ksize	
*
paddingVALID*
strides	
g
reshape_31/ShapeShape'average_pooling3d_31/AvgPool3D:output:0*
T0*
_output_shapes
:h
reshape_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_31/strided_sliceStridedSlicereshape_31/Shape:output:0'reshape_31/strided_slice/stack:output:0)reshape_31/strided_slice/stack_1:output:0)reshape_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
reshape_31/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������]
reshape_31/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_31/Reshape/shapePack!reshape_31/strided_slice:output:0#reshape_31/Reshape/shape/1:output:0#reshape_31/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
reshape_31/ReshapeReshape'average_pooling3d_31/AvgPool3D:output:0!reshape_31/Reshape/shape:output:0*
T0*,
_output_shapes
:���������(�o
IdentityIdentityreshape_31/Reshape:output:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp!^conv3d_93/BiasAdd/ReadVariableOp ^conv3d_93/Conv3D/ReadVariableOp!^conv3d_94/BiasAdd/ReadVariableOp ^conv3d_94/Conv3D/ReadVariableOp!^conv3d_95/BiasAdd/ReadVariableOp ^conv3d_95/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������
: : : : : : 2D
 conv3d_93/BiasAdd/ReadVariableOp conv3d_93/BiasAdd/ReadVariableOp2B
conv3d_93/Conv3D/ReadVariableOpconv3d_93/Conv3D/ReadVariableOp2D
 conv3d_94/BiasAdd/ReadVariableOp conv3d_94/BiasAdd/ReadVariableOp2B
conv3d_94/Conv3D/ReadVariableOpconv3d_94/Conv3D/ReadVariableOp2D
 conv3d_95/BiasAdd/ReadVariableOp conv3d_95/BiasAdd/ReadVariableOp2B
conv3d_95/Conv3D/ReadVariableOpconv3d_95/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������

 
_user_specified_namevideos
�
l
B__inference_add_10_layer_call_and_return_conditional_losses_383211

inputs
inputs_1
identityU
addAddV2inputsinputs_1*
T0*,
_output_shapes
:���������(�T
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:���������(�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������(�:���������(�:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs:TP
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
[
/__inference_concatenate_12_layer_call_fn_384805
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *S
fNRL
J__inference_concatenate_12_layer_call_and_return_conditional_losses_383102e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������(�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������(�:���������(�:V R
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/1
�-
�
P__inference_tubelet_embedding_30_layer_call_and_return_conditional_losses_384746

videosF
(conv3d_90_conv3d_readvariableop_resource: 7
)conv3d_90_biasadd_readvariableop_resource: F
(conv3d_91_conv3d_readvariableop_resource: @7
)conv3d_91_biasadd_readvariableop_resource:@G
(conv3d_92_conv3d_readvariableop_resource:@�8
)conv3d_92_biasadd_readvariableop_resource:	�
identity�� conv3d_90/BiasAdd/ReadVariableOp�conv3d_90/Conv3D/ReadVariableOp� conv3d_91/BiasAdd/ReadVariableOp�conv3d_91/Conv3D/ReadVariableOp� conv3d_92/BiasAdd/ReadVariableOp�conv3d_92/Conv3D/ReadVariableOp�
conv3d_90/Conv3D/ReadVariableOpReadVariableOp(conv3d_90_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
conv3d_90/Conv3DConv3Dvideos'conv3d_90/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
 conv3d_90/BiasAdd/ReadVariableOpReadVariableOp)conv3d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv3d_90/BiasAddBiasAddconv3d_90/Conv3D:output:0(conv3d_90/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 p
conv3d_90/ReluReluconv3d_90/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
max_pooling3d_60/MaxPool3D	MaxPool3Dconv3d_90/Relu:activations:0*
T0*3
_output_shapes!
:���������
 *
ksize	
*
paddingVALID*
strides	
�
conv3d_91/Conv3D/ReadVariableOpReadVariableOp(conv3d_91_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
conv3d_91/Conv3DConv3D#max_pooling3d_60/MaxPool3D:output:0'conv3d_91/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@*
paddingSAME*
strides	
�
 conv3d_91/BiasAdd/ReadVariableOpReadVariableOp)conv3d_91_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d_91/BiasAddBiasAddconv3d_91/Conv3D:output:0(conv3d_91/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@p
conv3d_91/ReluReluconv3d_91/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
@�
max_pooling3d_61/MaxPool3D	MaxPool3Dconv3d_91/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
conv3d_92/Conv3D/ReadVariableOpReadVariableOp(conv3d_92_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
conv3d_92/Conv3DConv3D#max_pooling3d_61/MaxPool3D:output:0'conv3d_92/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
 conv3d_92/BiasAdd/ReadVariableOpReadVariableOp)conv3d_92_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_92/BiasAddBiasAddconv3d_92/Conv3D:output:0(conv3d_92/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
average_pooling3d_30/AvgPool3D	AvgPool3Dconv3d_92/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
�*
ksize	
*
paddingVALID*
strides	
g
reshape_30/ShapeShape'average_pooling3d_30/AvgPool3D:output:0*
T0*
_output_shapes
:h
reshape_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_30/strided_sliceStridedSlicereshape_30/Shape:output:0'reshape_30/strided_slice/stack:output:0)reshape_30/strided_slice/stack_1:output:0)reshape_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
reshape_30/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������]
reshape_30/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_30/Reshape/shapePack!reshape_30/strided_slice:output:0#reshape_30/Reshape/shape/1:output:0#reshape_30/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
reshape_30/ReshapeReshape'average_pooling3d_30/AvgPool3D:output:0!reshape_30/Reshape/shape:output:0*
T0*,
_output_shapes
:���������(�o
IdentityIdentityreshape_30/Reshape:output:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp!^conv3d_90/BiasAdd/ReadVariableOp ^conv3d_90/Conv3D/ReadVariableOp!^conv3d_91/BiasAdd/ReadVariableOp ^conv3d_91/Conv3D/ReadVariableOp!^conv3d_92/BiasAdd/ReadVariableOp ^conv3d_92/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������
: : : : : : 2D
 conv3d_90/BiasAdd/ReadVariableOp conv3d_90/BiasAdd/ReadVariableOp2B
conv3d_90/Conv3D/ReadVariableOpconv3d_90/Conv3D/ReadVariableOp2D
 conv3d_91/BiasAdd/ReadVariableOp conv3d_91/BiasAdd/ReadVariableOp2B
conv3d_91/Conv3D/ReadVariableOpconv3d_91/Conv3D/ReadVariableOp2D
 conv3d_92/BiasAdd/ReadVariableOp conv3d_92/BiasAdd/ReadVariableOp2B
conv3d_92/Conv3D/ReadVariableOpconv3d_92/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������

 
_user_specified_namevideos
�[
�
C__inference_model_5_layer_call_and_return_conditional_losses_383727

inputs9
tubelet_embedding_30_383648: )
tubelet_embedding_30_383650: 9
tubelet_embedding_30_383652: @)
tubelet_embedding_30_383654:@:
tubelet_embedding_30_383656:@�*
tubelet_embedding_30_383658:	�9
tubelet_embedding_31_383661: )
tubelet_embedding_31_383663: 9
tubelet_embedding_31_383665: @)
tubelet_embedding_31_383667:@:
tubelet_embedding_31_383669:@�*
tubelet_embedding_31_383671:	� 
positional_encoder_15_383675/
positional_encoder_15_383677:	(�,
layer_normalization_15_383680:	�,
layer_normalization_15_383682:	�5
multi_head_attention_5_383685:��0
multi_head_attention_5_383687:	�5
multi_head_attention_5_383689:��0
multi_head_attention_5_383691:	�5
multi_head_attention_5_383693:��0
multi_head_attention_5_383695:	�5
multi_head_attention_5_383697:��,
multi_head_attention_5_383699:	�,
layer_normalization_16_383704:	�,
layer_normalization_16_383706:	�'
sequential_5_383709:
��"
sequential_5_383711:	�,
layer_normalization_17_383715:	�,
layer_normalization_17_383717:	�"
dense_11_383721:	�
dense_11_383723:
identity�� dense_11/StatefulPartitionedCall�.layer_normalization_15/StatefulPartitionedCall�.layer_normalization_16/StatefulPartitionedCall�.layer_normalization_17/StatefulPartitionedCall�.multi_head_attention_5/StatefulPartitionedCall�-positional_encoder_15/StatefulPartitionedCall�$sequential_5/StatefulPartitionedCall�,tubelet_embedding_30/StatefulPartitionedCall�,tubelet_embedding_31/StatefulPartitionedCall�
/tf.__operators__.getitem_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_35/strided_sliceStridedSliceinputs8tf.__operators__.getitem_35/strided_slice/stack:output:0:tf.__operators__.getitem_35/strided_slice/stack_1:output:0:tf.__operators__.getitem_35/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
/tf.__operators__.getitem_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_34/strided_sliceStridedSliceinputs8tf.__operators__.getitem_34/strided_slice/stack:output:0:tf.__operators__.getitem_34/strided_slice/stack_1:output:0:tf.__operators__.getitem_34/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
,tubelet_embedding_30/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_34/strided_slice:output:0tubelet_embedding_30_383648tubelet_embedding_30_383650tubelet_embedding_30_383652tubelet_embedding_30_383654tubelet_embedding_30_383656tubelet_embedding_30_383658*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_tubelet_embedding_30_layer_call_and_return_conditional_losses_383031�
,tubelet_embedding_31/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_35/strided_slice:output:0tubelet_embedding_31_383661tubelet_embedding_31_383663tubelet_embedding_31_383665tubelet_embedding_31_383667tubelet_embedding_31_383669tubelet_embedding_31_383671*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_tubelet_embedding_31_layer_call_and_return_conditional_losses_383081�
concatenate_12/PartitionedCallPartitionedCall5tubelet_embedding_30/StatefulPartitionedCall:output:05tubelet_embedding_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *S
fNRL
J__inference_concatenate_12_layer_call_and_return_conditional_losses_383102�
-positional_encoder_15/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0positional_encoder_15_383675positional_encoder_15_383677*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Z
fURS
Q__inference_positional_encoder_15_layer_call_and_return_conditional_losses_383116�
.layer_normalization_15/StatefulPartitionedCallStatefulPartitionedCall6positional_encoder_15/StatefulPartitionedCall:output:0layer_normalization_15_383680layer_normalization_15_383682*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_383144�
.multi_head_attention_5/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_15/StatefulPartitionedCall:output:07layer_normalization_15/StatefulPartitionedCall:output:0multi_head_attention_5_383685multi_head_attention_5_383687multi_head_attention_5_383689multi_head_attention_5_383691multi_head_attention_5_383693multi_head_attention_5_383695multi_head_attention_5_383697multi_head_attention_5_383699*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:���������(�:���������((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_383482�
add_10/PartitionedCallPartitionedCall7multi_head_attention_5/StatefulPartitionedCall:output:06positional_encoder_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_add_10_layer_call_and_return_conditional_losses_383211�
.layer_normalization_16/StatefulPartitionedCallStatefulPartitionedCalladd_10/PartitionedCall:output:0layer_normalization_16_383704layer_normalization_16_383706*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_16_layer_call_and_return_conditional_losses_383235�
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_16/StatefulPartitionedCall:output:0sequential_5_383709sequential_5_383711*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_382933�
add_11/PartitionedCallPartitionedCall-sequential_5/StatefulPartitionedCall:output:0add_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_add_11_layer_call_and_return_conditional_losses_383252�
.layer_normalization_17/StatefulPartitionedCallStatefulPartitionedCalladd_11/PartitionedCall:output:0layer_normalization_17_383715layer_normalization_17_383717*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_17_layer_call_and_return_conditional_losses_383276�
*global_average_pooling1d_5/PartitionedCallPartitionedCall7layer_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *_
fZRX
V__inference_global_average_pooling1d_5_layer_call_and_return_conditional_losses_382977�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_5/PartitionedCall:output:0dense_11_383721dense_11_383723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_383293x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_11/StatefulPartitionedCall/^layer_normalization_15/StatefulPartitionedCall/^layer_normalization_16/StatefulPartitionedCall/^layer_normalization_17/StatefulPartitionedCall/^multi_head_attention_5/StatefulPartitionedCall.^positional_encoder_15/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall-^tubelet_embedding_30/StatefulPartitionedCall-^tubelet_embedding_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2`
.layer_normalization_15/StatefulPartitionedCall.layer_normalization_15/StatefulPartitionedCall2`
.layer_normalization_16/StatefulPartitionedCall.layer_normalization_16/StatefulPartitionedCall2`
.layer_normalization_17/StatefulPartitionedCall.layer_normalization_17/StatefulPartitionedCall2`
.multi_head_attention_5/StatefulPartitionedCall.multi_head_attention_5/StatefulPartitionedCall2^
-positional_encoder_15/StatefulPartitionedCall-positional_encoder_15/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2\
,tubelet_embedding_30/StatefulPartitionedCall,tubelet_embedding_30/StatefulPartitionedCall2\
,tubelet_embedding_31/StatefulPartitionedCall,tubelet_embedding_31/StatefulPartitionedCall:[ W
3
_output_shapes!
:���������

 
_user_specified_nameinputs: 

_output_shapes
:(
�
l
B__inference_add_11_layer_call_and_return_conditional_losses_383252

inputs
inputs_1
identityU
addAddV2inputsinputs_1*
T0*,
_output_shapes
:���������(�T
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:���������(�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������(�:���������(�:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs:TP
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�	
�
5__inference_tubelet_embedding_30_layer_call_fn_384710

videos%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvideosunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_tubelet_embedding_30_layer_call_and_return_conditional_losses_383031t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������

 
_user_specified_namevideos
�	
�
D__inference_dense_11_layer_call_and_return_conditional_losses_383293

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_layer_normalization_17_layer_call_and_return_conditional_losses_383276

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������(�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������(�l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�h
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(�g
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
�
R__inference_layer_normalization_17_layer_call_and_return_conditional_losses_385171

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������(�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������(�l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�h
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(�g
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
S
'__inference_add_10_layer_call_fn_384997
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_add_10_layer_call_and_return_conditional_losses_383211e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������(�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������(�:���������(�:V R
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/1
�
M
1__inference_max_pooling3d_62_layer_call_fn_385236

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_max_pooling3d_62_layer_call_and_return_conditional_losses_382817�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
��
�M
"__inference__traced_restore_385954
file_prefix<
-assignvariableop_layer_normalization_15_gamma:	�=
.assignvariableop_1_layer_normalization_15_beta:	�>
/assignvariableop_2_layer_normalization_16_gamma:	�=
.assignvariableop_3_layer_normalization_16_beta:	�>
/assignvariableop_4_layer_normalization_17_gamma:	�=
.assignvariableop_5_layer_normalization_17_beta:	�5
"assignvariableop_6_dense_11_kernel:	�.
 assignvariableop_7_dense_11_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: W
9assignvariableop_13_tubelet_embedding_30_conv3d_90_kernel: E
7assignvariableop_14_tubelet_embedding_30_conv3d_90_bias: W
9assignvariableop_15_tubelet_embedding_30_conv3d_91_kernel: @E
7assignvariableop_16_tubelet_embedding_30_conv3d_91_bias:@X
9assignvariableop_17_tubelet_embedding_30_conv3d_92_kernel:@�F
7assignvariableop_18_tubelet_embedding_30_conv3d_92_bias:	�W
9assignvariableop_19_tubelet_embedding_31_conv3d_93_kernel: E
7assignvariableop_20_tubelet_embedding_31_conv3d_93_bias: W
9assignvariableop_21_tubelet_embedding_31_conv3d_94_kernel: @E
7assignvariableop_22_tubelet_embedding_31_conv3d_94_bias:@X
9assignvariableop_23_tubelet_embedding_31_conv3d_95_kernel:@�F
7assignvariableop_24_tubelet_embedding_31_conv3d_95_bias:	�Q
>assignvariableop_25_positional_encoder_15_embedding_embeddings:	(�O
7assignvariableop_26_multi_head_attention_5_query_kernel:��H
5assignvariableop_27_multi_head_attention_5_query_bias:	�M
5assignvariableop_28_multi_head_attention_5_key_kernel:��F
3assignvariableop_29_multi_head_attention_5_key_bias:	�O
7assignvariableop_30_multi_head_attention_5_value_kernel:��H
5assignvariableop_31_multi_head_attention_5_value_bias:	�Z
Bassignvariableop_32_multi_head_attention_5_attention_output_kernel:��O
@assignvariableop_33_multi_head_attention_5_attention_output_bias:	�7
#assignvariableop_34_dense_10_kernel:
��0
!assignvariableop_35_dense_10_bias:	�#
assignvariableop_36_total: #
assignvariableop_37_count: %
assignvariableop_38_total_1: %
assignvariableop_39_count_1: F
7assignvariableop_40_adam_layer_normalization_15_gamma_m:	�E
6assignvariableop_41_adam_layer_normalization_15_beta_m:	�F
7assignvariableop_42_adam_layer_normalization_16_gamma_m:	�E
6assignvariableop_43_adam_layer_normalization_16_beta_m:	�F
7assignvariableop_44_adam_layer_normalization_17_gamma_m:	�E
6assignvariableop_45_adam_layer_normalization_17_beta_m:	�=
*assignvariableop_46_adam_dense_11_kernel_m:	�6
(assignvariableop_47_adam_dense_11_bias_m:^
@assignvariableop_48_adam_tubelet_embedding_30_conv3d_90_kernel_m: L
>assignvariableop_49_adam_tubelet_embedding_30_conv3d_90_bias_m: ^
@assignvariableop_50_adam_tubelet_embedding_30_conv3d_91_kernel_m: @L
>assignvariableop_51_adam_tubelet_embedding_30_conv3d_91_bias_m:@_
@assignvariableop_52_adam_tubelet_embedding_30_conv3d_92_kernel_m:@�M
>assignvariableop_53_adam_tubelet_embedding_30_conv3d_92_bias_m:	�^
@assignvariableop_54_adam_tubelet_embedding_31_conv3d_93_kernel_m: L
>assignvariableop_55_adam_tubelet_embedding_31_conv3d_93_bias_m: ^
@assignvariableop_56_adam_tubelet_embedding_31_conv3d_94_kernel_m: @L
>assignvariableop_57_adam_tubelet_embedding_31_conv3d_94_bias_m:@_
@assignvariableop_58_adam_tubelet_embedding_31_conv3d_95_kernel_m:@�M
>assignvariableop_59_adam_tubelet_embedding_31_conv3d_95_bias_m:	�X
Eassignvariableop_60_adam_positional_encoder_15_embedding_embeddings_m:	(�V
>assignvariableop_61_adam_multi_head_attention_5_query_kernel_m:��O
<assignvariableop_62_adam_multi_head_attention_5_query_bias_m:	�T
<assignvariableop_63_adam_multi_head_attention_5_key_kernel_m:��M
:assignvariableop_64_adam_multi_head_attention_5_key_bias_m:	�V
>assignvariableop_65_adam_multi_head_attention_5_value_kernel_m:��O
<assignvariableop_66_adam_multi_head_attention_5_value_bias_m:	�a
Iassignvariableop_67_adam_multi_head_attention_5_attention_output_kernel_m:��V
Gassignvariableop_68_adam_multi_head_attention_5_attention_output_bias_m:	�>
*assignvariableop_69_adam_dense_10_kernel_m:
��7
(assignvariableop_70_adam_dense_10_bias_m:	�F
7assignvariableop_71_adam_layer_normalization_15_gamma_v:	�E
6assignvariableop_72_adam_layer_normalization_15_beta_v:	�F
7assignvariableop_73_adam_layer_normalization_16_gamma_v:	�E
6assignvariableop_74_adam_layer_normalization_16_beta_v:	�F
7assignvariableop_75_adam_layer_normalization_17_gamma_v:	�E
6assignvariableop_76_adam_layer_normalization_17_beta_v:	�=
*assignvariableop_77_adam_dense_11_kernel_v:	�6
(assignvariableop_78_adam_dense_11_bias_v:^
@assignvariableop_79_adam_tubelet_embedding_30_conv3d_90_kernel_v: L
>assignvariableop_80_adam_tubelet_embedding_30_conv3d_90_bias_v: ^
@assignvariableop_81_adam_tubelet_embedding_30_conv3d_91_kernel_v: @L
>assignvariableop_82_adam_tubelet_embedding_30_conv3d_91_bias_v:@_
@assignvariableop_83_adam_tubelet_embedding_30_conv3d_92_kernel_v:@�M
>assignvariableop_84_adam_tubelet_embedding_30_conv3d_92_bias_v:	�^
@assignvariableop_85_adam_tubelet_embedding_31_conv3d_93_kernel_v: L
>assignvariableop_86_adam_tubelet_embedding_31_conv3d_93_bias_v: ^
@assignvariableop_87_adam_tubelet_embedding_31_conv3d_94_kernel_v: @L
>assignvariableop_88_adam_tubelet_embedding_31_conv3d_94_bias_v:@_
@assignvariableop_89_adam_tubelet_embedding_31_conv3d_95_kernel_v:@�M
>assignvariableop_90_adam_tubelet_embedding_31_conv3d_95_bias_v:	�X
Eassignvariableop_91_adam_positional_encoder_15_embedding_embeddings_v:	(�V
>assignvariableop_92_adam_multi_head_attention_5_query_kernel_v:��O
<assignvariableop_93_adam_multi_head_attention_5_query_bias_v:	�T
<assignvariableop_94_adam_multi_head_attention_5_key_kernel_v:��M
:assignvariableop_95_adam_multi_head_attention_5_key_bias_v:	�V
>assignvariableop_96_adam_multi_head_attention_5_value_kernel_v:��O
<assignvariableop_97_adam_multi_head_attention_5_value_bias_v:	�a
Iassignvariableop_98_adam_multi_head_attention_5_attention_output_kernel_v:��V
Gassignvariableop_99_adam_multi_head_attention_5_attention_output_bias_v:	�?
+assignvariableop_100_adam_dense_10_kernel_v:
��8
)assignvariableop_101_adam_dense_10_bias_v:	�
identity_103��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�2
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:g*
dtype0*�1
value�1B�1gB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:g*
dtype0*�
value�B�gB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*u
dtypesk
i2g	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp-assignvariableop_layer_normalization_15_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_layer_normalization_15_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_layer_normalization_16_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_layer_normalization_16_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp/assignvariableop_4_layer_normalization_17_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_layer_normalization_17_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp9assignvariableop_13_tubelet_embedding_30_conv3d_90_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp7assignvariableop_14_tubelet_embedding_30_conv3d_90_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp9assignvariableop_15_tubelet_embedding_30_conv3d_91_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp7assignvariableop_16_tubelet_embedding_30_conv3d_91_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_tubelet_embedding_30_conv3d_92_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp7assignvariableop_18_tubelet_embedding_30_conv3d_92_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp9assignvariableop_19_tubelet_embedding_31_conv3d_93_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp7assignvariableop_20_tubelet_embedding_31_conv3d_93_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp9assignvariableop_21_tubelet_embedding_31_conv3d_94_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp7assignvariableop_22_tubelet_embedding_31_conv3d_94_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp9assignvariableop_23_tubelet_embedding_31_conv3d_95_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp7assignvariableop_24_tubelet_embedding_31_conv3d_95_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp>assignvariableop_25_positional_encoder_15_embedding_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp7assignvariableop_26_multi_head_attention_5_query_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp5assignvariableop_27_multi_head_attention_5_query_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_multi_head_attention_5_key_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp3assignvariableop_29_multi_head_attention_5_key_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp7assignvariableop_30_multi_head_attention_5_value_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp5assignvariableop_31_multi_head_attention_5_value_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpBassignvariableop_32_multi_head_attention_5_attention_output_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp@assignvariableop_33_multi_head_attention_5_attention_output_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_10_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_10_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_totalIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_countIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_count_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_layer_normalization_15_gamma_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_layer_normalization_15_beta_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp7assignvariableop_42_adam_layer_normalization_16_gamma_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_layer_normalization_16_beta_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adam_layer_normalization_17_gamma_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_layer_normalization_17_beta_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dense_11_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense_11_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp@assignvariableop_48_adam_tubelet_embedding_30_conv3d_90_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp>assignvariableop_49_adam_tubelet_embedding_30_conv3d_90_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp@assignvariableop_50_adam_tubelet_embedding_30_conv3d_91_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp>assignvariableop_51_adam_tubelet_embedding_30_conv3d_91_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp@assignvariableop_52_adam_tubelet_embedding_30_conv3d_92_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp>assignvariableop_53_adam_tubelet_embedding_30_conv3d_92_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp@assignvariableop_54_adam_tubelet_embedding_31_conv3d_93_kernel_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp>assignvariableop_55_adam_tubelet_embedding_31_conv3d_93_bias_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp@assignvariableop_56_adam_tubelet_embedding_31_conv3d_94_kernel_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp>assignvariableop_57_adam_tubelet_embedding_31_conv3d_94_bias_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp@assignvariableop_58_adam_tubelet_embedding_31_conv3d_95_kernel_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp>assignvariableop_59_adam_tubelet_embedding_31_conv3d_95_bias_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpEassignvariableop_60_adam_positional_encoder_15_embedding_embeddings_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp>assignvariableop_61_adam_multi_head_attention_5_query_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp<assignvariableop_62_adam_multi_head_attention_5_query_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp<assignvariableop_63_adam_multi_head_attention_5_key_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp:assignvariableop_64_adam_multi_head_attention_5_key_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp>assignvariableop_65_adam_multi_head_attention_5_value_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp<assignvariableop_66_adam_multi_head_attention_5_value_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpIassignvariableop_67_adam_multi_head_attention_5_attention_output_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpGassignvariableop_68_adam_multi_head_attention_5_attention_output_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dense_10_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_dense_10_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_layer_normalization_15_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_layer_normalization_15_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp7assignvariableop_73_adam_layer_normalization_16_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_layer_normalization_16_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_layer_normalization_17_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_layer_normalization_17_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_11_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_11_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp@assignvariableop_79_adam_tubelet_embedding_30_conv3d_90_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp>assignvariableop_80_adam_tubelet_embedding_30_conv3d_90_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp@assignvariableop_81_adam_tubelet_embedding_30_conv3d_91_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp>assignvariableop_82_adam_tubelet_embedding_30_conv3d_91_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp@assignvariableop_83_adam_tubelet_embedding_30_conv3d_92_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp>assignvariableop_84_adam_tubelet_embedding_30_conv3d_92_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp@assignvariableop_85_adam_tubelet_embedding_31_conv3d_93_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp>assignvariableop_86_adam_tubelet_embedding_31_conv3d_93_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp@assignvariableop_87_adam_tubelet_embedding_31_conv3d_94_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp>assignvariableop_88_adam_tubelet_embedding_31_conv3d_94_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp@assignvariableop_89_adam_tubelet_embedding_31_conv3d_95_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp>assignvariableop_90_adam_tubelet_embedding_31_conv3d_95_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOpEassignvariableop_91_adam_positional_encoder_15_embedding_embeddings_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp>assignvariableop_92_adam_multi_head_attention_5_query_kernel_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp<assignvariableop_93_adam_multi_head_attention_5_query_bias_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp<assignvariableop_94_adam_multi_head_attention_5_key_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp:assignvariableop_95_adam_multi_head_attention_5_key_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp>assignvariableop_96_adam_multi_head_attention_5_value_kernel_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp<assignvariableop_97_adam_multi_head_attention_5_value_bias_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOpIassignvariableop_98_adam_multi_head_attention_5_attention_output_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOpGassignvariableop_99_adam_multi_head_attention_5_attention_output_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp+assignvariableop_100_adam_dense_10_kernel_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp)assignvariableop_101_adam_dense_10_bias_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_102Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_103IdentityIdentity_102:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_103Identity_103:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
�	
�
D__inference_dense_11_layer_call_and_return_conditional_losses_385201

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_model_5_layer_call_fn_384118

inputs%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�'
	unknown_5: 
	unknown_6: '
	unknown_7: @
	unknown_8:@(
	unknown_9:@�

unknown_10:	�

unknown_11

unknown_12:	(�

unknown_13:	�

unknown_14:	�"

unknown_15:��

unknown_16:	�"

unknown_17:��

unknown_18:	�"

unknown_19:��

unknown_20:	�"

unknown_21:��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�

unknown_30:
identity��StatefulPartitionedCall�
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
:���������*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_383300o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������

 
_user_specified_nameinputs: 

_output_shapes
:(
�
�
-__inference_sequential_5_layer_call_fn_382903
dense_10_input
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_10_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_382896t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:���������(�
(
_user_specified_namedense_10_input
��
�6
__inference__traced_save_385638
file_prefix;
7savev2_layer_normalization_15_gamma_read_readvariableop:
6savev2_layer_normalization_15_beta_read_readvariableop;
7savev2_layer_normalization_16_gamma_read_readvariableop:
6savev2_layer_normalization_16_beta_read_readvariableop;
7savev2_layer_normalization_17_gamma_read_readvariableop:
6savev2_layer_normalization_17_beta_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_tubelet_embedding_30_conv3d_90_kernel_read_readvariableopB
>savev2_tubelet_embedding_30_conv3d_90_bias_read_readvariableopD
@savev2_tubelet_embedding_30_conv3d_91_kernel_read_readvariableopB
>savev2_tubelet_embedding_30_conv3d_91_bias_read_readvariableopD
@savev2_tubelet_embedding_30_conv3d_92_kernel_read_readvariableopB
>savev2_tubelet_embedding_30_conv3d_92_bias_read_readvariableopD
@savev2_tubelet_embedding_31_conv3d_93_kernel_read_readvariableopB
>savev2_tubelet_embedding_31_conv3d_93_bias_read_readvariableopD
@savev2_tubelet_embedding_31_conv3d_94_kernel_read_readvariableopB
>savev2_tubelet_embedding_31_conv3d_94_bias_read_readvariableopD
@savev2_tubelet_embedding_31_conv3d_95_kernel_read_readvariableopB
>savev2_tubelet_embedding_31_conv3d_95_bias_read_readvariableopI
Esavev2_positional_encoder_15_embedding_embeddings_read_readvariableopB
>savev2_multi_head_attention_5_query_kernel_read_readvariableop@
<savev2_multi_head_attention_5_query_bias_read_readvariableop@
<savev2_multi_head_attention_5_key_kernel_read_readvariableop>
:savev2_multi_head_attention_5_key_bias_read_readvariableopB
>savev2_multi_head_attention_5_value_kernel_read_readvariableop@
<savev2_multi_head_attention_5_value_bias_read_readvariableopM
Isavev2_multi_head_attention_5_attention_output_kernel_read_readvariableopK
Gsavev2_multi_head_attention_5_attention_output_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopB
>savev2_adam_layer_normalization_15_gamma_m_read_readvariableopA
=savev2_adam_layer_normalization_15_beta_m_read_readvariableopB
>savev2_adam_layer_normalization_16_gamma_m_read_readvariableopA
=savev2_adam_layer_normalization_16_beta_m_read_readvariableopB
>savev2_adam_layer_normalization_17_gamma_m_read_readvariableopA
=savev2_adam_layer_normalization_17_beta_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableopK
Gsavev2_adam_tubelet_embedding_30_conv3d_90_kernel_m_read_readvariableopI
Esavev2_adam_tubelet_embedding_30_conv3d_90_bias_m_read_readvariableopK
Gsavev2_adam_tubelet_embedding_30_conv3d_91_kernel_m_read_readvariableopI
Esavev2_adam_tubelet_embedding_30_conv3d_91_bias_m_read_readvariableopK
Gsavev2_adam_tubelet_embedding_30_conv3d_92_kernel_m_read_readvariableopI
Esavev2_adam_tubelet_embedding_30_conv3d_92_bias_m_read_readvariableopK
Gsavev2_adam_tubelet_embedding_31_conv3d_93_kernel_m_read_readvariableopI
Esavev2_adam_tubelet_embedding_31_conv3d_93_bias_m_read_readvariableopK
Gsavev2_adam_tubelet_embedding_31_conv3d_94_kernel_m_read_readvariableopI
Esavev2_adam_tubelet_embedding_31_conv3d_94_bias_m_read_readvariableopK
Gsavev2_adam_tubelet_embedding_31_conv3d_95_kernel_m_read_readvariableopI
Esavev2_adam_tubelet_embedding_31_conv3d_95_bias_m_read_readvariableopP
Lsavev2_adam_positional_encoder_15_embedding_embeddings_m_read_readvariableopI
Esavev2_adam_multi_head_attention_5_query_kernel_m_read_readvariableopG
Csavev2_adam_multi_head_attention_5_query_bias_m_read_readvariableopG
Csavev2_adam_multi_head_attention_5_key_kernel_m_read_readvariableopE
Asavev2_adam_multi_head_attention_5_key_bias_m_read_readvariableopI
Esavev2_adam_multi_head_attention_5_value_kernel_m_read_readvariableopG
Csavev2_adam_multi_head_attention_5_value_bias_m_read_readvariableopT
Psavev2_adam_multi_head_attention_5_attention_output_kernel_m_read_readvariableopR
Nsavev2_adam_multi_head_attention_5_attention_output_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableopB
>savev2_adam_layer_normalization_15_gamma_v_read_readvariableopA
=savev2_adam_layer_normalization_15_beta_v_read_readvariableopB
>savev2_adam_layer_normalization_16_gamma_v_read_readvariableopA
=savev2_adam_layer_normalization_16_beta_v_read_readvariableopB
>savev2_adam_layer_normalization_17_gamma_v_read_readvariableopA
=savev2_adam_layer_normalization_17_beta_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableopK
Gsavev2_adam_tubelet_embedding_30_conv3d_90_kernel_v_read_readvariableopI
Esavev2_adam_tubelet_embedding_30_conv3d_90_bias_v_read_readvariableopK
Gsavev2_adam_tubelet_embedding_30_conv3d_91_kernel_v_read_readvariableopI
Esavev2_adam_tubelet_embedding_30_conv3d_91_bias_v_read_readvariableopK
Gsavev2_adam_tubelet_embedding_30_conv3d_92_kernel_v_read_readvariableopI
Esavev2_adam_tubelet_embedding_30_conv3d_92_bias_v_read_readvariableopK
Gsavev2_adam_tubelet_embedding_31_conv3d_93_kernel_v_read_readvariableopI
Esavev2_adam_tubelet_embedding_31_conv3d_93_bias_v_read_readvariableopK
Gsavev2_adam_tubelet_embedding_31_conv3d_94_kernel_v_read_readvariableopI
Esavev2_adam_tubelet_embedding_31_conv3d_94_bias_v_read_readvariableopK
Gsavev2_adam_tubelet_embedding_31_conv3d_95_kernel_v_read_readvariableopI
Esavev2_adam_tubelet_embedding_31_conv3d_95_bias_v_read_readvariableopP
Lsavev2_adam_positional_encoder_15_embedding_embeddings_v_read_readvariableopI
Esavev2_adam_multi_head_attention_5_query_kernel_v_read_readvariableopG
Csavev2_adam_multi_head_attention_5_query_bias_v_read_readvariableopG
Csavev2_adam_multi_head_attention_5_key_kernel_v_read_readvariableopE
Asavev2_adam_multi_head_attention_5_key_bias_v_read_readvariableopI
Esavev2_adam_multi_head_attention_5_value_kernel_v_read_readvariableopG
Csavev2_adam_multi_head_attention_5_value_bias_v_read_readvariableopT
Psavev2_adam_multi_head_attention_5_attention_output_kernel_v_read_readvariableopR
Nsavev2_adam_multi_head_attention_5_attention_output_bias_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop
savev2_const_1

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �2
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:g*
dtype0*�1
value�1B�1gB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:g*
dtype0*�
value�B�gB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �4
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_layer_normalization_15_gamma_read_readvariableop6savev2_layer_normalization_15_beta_read_readvariableop7savev2_layer_normalization_16_gamma_read_readvariableop6savev2_layer_normalization_16_beta_read_readvariableop7savev2_layer_normalization_17_gamma_read_readvariableop6savev2_layer_normalization_17_beta_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_tubelet_embedding_30_conv3d_90_kernel_read_readvariableop>savev2_tubelet_embedding_30_conv3d_90_bias_read_readvariableop@savev2_tubelet_embedding_30_conv3d_91_kernel_read_readvariableop>savev2_tubelet_embedding_30_conv3d_91_bias_read_readvariableop@savev2_tubelet_embedding_30_conv3d_92_kernel_read_readvariableop>savev2_tubelet_embedding_30_conv3d_92_bias_read_readvariableop@savev2_tubelet_embedding_31_conv3d_93_kernel_read_readvariableop>savev2_tubelet_embedding_31_conv3d_93_bias_read_readvariableop@savev2_tubelet_embedding_31_conv3d_94_kernel_read_readvariableop>savev2_tubelet_embedding_31_conv3d_94_bias_read_readvariableop@savev2_tubelet_embedding_31_conv3d_95_kernel_read_readvariableop>savev2_tubelet_embedding_31_conv3d_95_bias_read_readvariableopEsavev2_positional_encoder_15_embedding_embeddings_read_readvariableop>savev2_multi_head_attention_5_query_kernel_read_readvariableop<savev2_multi_head_attention_5_query_bias_read_readvariableop<savev2_multi_head_attention_5_key_kernel_read_readvariableop:savev2_multi_head_attention_5_key_bias_read_readvariableop>savev2_multi_head_attention_5_value_kernel_read_readvariableop<savev2_multi_head_attention_5_value_bias_read_readvariableopIsavev2_multi_head_attention_5_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_5_attention_output_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adam_layer_normalization_15_gamma_m_read_readvariableop=savev2_adam_layer_normalization_15_beta_m_read_readvariableop>savev2_adam_layer_normalization_16_gamma_m_read_readvariableop=savev2_adam_layer_normalization_16_beta_m_read_readvariableop>savev2_adam_layer_normalization_17_gamma_m_read_readvariableop=savev2_adam_layer_normalization_17_beta_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableopGsavev2_adam_tubelet_embedding_30_conv3d_90_kernel_m_read_readvariableopEsavev2_adam_tubelet_embedding_30_conv3d_90_bias_m_read_readvariableopGsavev2_adam_tubelet_embedding_30_conv3d_91_kernel_m_read_readvariableopEsavev2_adam_tubelet_embedding_30_conv3d_91_bias_m_read_readvariableopGsavev2_adam_tubelet_embedding_30_conv3d_92_kernel_m_read_readvariableopEsavev2_adam_tubelet_embedding_30_conv3d_92_bias_m_read_readvariableopGsavev2_adam_tubelet_embedding_31_conv3d_93_kernel_m_read_readvariableopEsavev2_adam_tubelet_embedding_31_conv3d_93_bias_m_read_readvariableopGsavev2_adam_tubelet_embedding_31_conv3d_94_kernel_m_read_readvariableopEsavev2_adam_tubelet_embedding_31_conv3d_94_bias_m_read_readvariableopGsavev2_adam_tubelet_embedding_31_conv3d_95_kernel_m_read_readvariableopEsavev2_adam_tubelet_embedding_31_conv3d_95_bias_m_read_readvariableopLsavev2_adam_positional_encoder_15_embedding_embeddings_m_read_readvariableopEsavev2_adam_multi_head_attention_5_query_kernel_m_read_readvariableopCsavev2_adam_multi_head_attention_5_query_bias_m_read_readvariableopCsavev2_adam_multi_head_attention_5_key_kernel_m_read_readvariableopAsavev2_adam_multi_head_attention_5_key_bias_m_read_readvariableopEsavev2_adam_multi_head_attention_5_value_kernel_m_read_readvariableopCsavev2_adam_multi_head_attention_5_value_bias_m_read_readvariableopPsavev2_adam_multi_head_attention_5_attention_output_kernel_m_read_readvariableopNsavev2_adam_multi_head_attention_5_attention_output_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop>savev2_adam_layer_normalization_15_gamma_v_read_readvariableop=savev2_adam_layer_normalization_15_beta_v_read_readvariableop>savev2_adam_layer_normalization_16_gamma_v_read_readvariableop=savev2_adam_layer_normalization_16_beta_v_read_readvariableop>savev2_adam_layer_normalization_17_gamma_v_read_readvariableop=savev2_adam_layer_normalization_17_beta_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableopGsavev2_adam_tubelet_embedding_30_conv3d_90_kernel_v_read_readvariableopEsavev2_adam_tubelet_embedding_30_conv3d_90_bias_v_read_readvariableopGsavev2_adam_tubelet_embedding_30_conv3d_91_kernel_v_read_readvariableopEsavev2_adam_tubelet_embedding_30_conv3d_91_bias_v_read_readvariableopGsavev2_adam_tubelet_embedding_30_conv3d_92_kernel_v_read_readvariableopEsavev2_adam_tubelet_embedding_30_conv3d_92_bias_v_read_readvariableopGsavev2_adam_tubelet_embedding_31_conv3d_93_kernel_v_read_readvariableopEsavev2_adam_tubelet_embedding_31_conv3d_93_bias_v_read_readvariableopGsavev2_adam_tubelet_embedding_31_conv3d_94_kernel_v_read_readvariableopEsavev2_adam_tubelet_embedding_31_conv3d_94_bias_v_read_readvariableopGsavev2_adam_tubelet_embedding_31_conv3d_95_kernel_v_read_readvariableopEsavev2_adam_tubelet_embedding_31_conv3d_95_bias_v_read_readvariableopLsavev2_adam_positional_encoder_15_embedding_embeddings_v_read_readvariableopEsavev2_adam_multi_head_attention_5_query_kernel_v_read_readvariableopCsavev2_adam_multi_head_attention_5_query_bias_v_read_readvariableopCsavev2_adam_multi_head_attention_5_key_kernel_v_read_readvariableopAsavev2_adam_multi_head_attention_5_key_bias_v_read_readvariableopEsavev2_adam_multi_head_attention_5_value_kernel_v_read_readvariableopCsavev2_adam_multi_head_attention_5_value_bias_v_read_readvariableopPsavev2_adam_multi_head_attention_5_attention_output_kernel_v_read_readvariableopNsavev2_adam_multi_head_attention_5_attention_output_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *u
dtypesk
i2g	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�:�:�:�:�:	�:: : : : : : : : @:@:@�:�: : : @:@:@�:�:	(�:��:	�:��:	�:��:	�:��:�:
��:�: : : : :�:�:�:�:�:�:	�:: : : @:@:@�:�: : : @:@:@�:�:	(�:��:	�:��:	�:��:	�:��:�:
��:�:�:�:�:�:�:�:	�:: : : @:@:@�:�: : : @:@:@�:�:	(�:��:	�:��:	�:��:	�:��:�:
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 
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
:@�:!

_output_shapes	
:�:0,
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
:@�:!

_output_shapes	
:�:%!

_output_shapes
:	(�:*&
$
_output_shapes
:��:%!

_output_shapes
:	�:*&
$
_output_shapes
:��:%!

_output_shapes
:	�:*&
$
_output_shapes
:��:% !

_output_shapes
:	�:*!&
$
_output_shapes
:��:!"

_output_shapes	
:�:&#"
 
_output_shapes
:
��:!$

_output_shapes	
:�:%
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
:�:!*

_output_shapes	
:�:!+

_output_shapes	
:�:!,

_output_shapes	
:�:!-

_output_shapes	
:�:!.

_output_shapes	
:�:%/!

_output_shapes
:	�: 0
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
:@�:!6

_output_shapes	
:�:07,
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
:@�:!<

_output_shapes	
:�:%=!

_output_shapes
:	(�:*>&
$
_output_shapes
:��:%?!

_output_shapes
:	�:*@&
$
_output_shapes
:��:%A!

_output_shapes
:	�:*B&
$
_output_shapes
:��:%C!

_output_shapes
:	�:*D&
$
_output_shapes
:��:!E

_output_shapes	
:�:&F"
 
_output_shapes
:
��:!G

_output_shapes	
:�:!H

_output_shapes	
:�:!I

_output_shapes	
:�:!J

_output_shapes	
:�:!K

_output_shapes	
:�:!L

_output_shapes	
:�:!M

_output_shapes	
:�:%N!

_output_shapes
:	�: O
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
:@�:!U

_output_shapes	
:�:0V,
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
:@�:![

_output_shapes	
:�:%\!

_output_shapes
:	(�:*]&
$
_output_shapes
:��:%^!

_output_shapes
:	�:*_&
$
_output_shapes
:��:%`!

_output_shapes
:	�:*a&
$
_output_shapes
:��:%b!

_output_shapes
:	�:*c&
$
_output_shapes
:��:!d

_output_shapes	
:�:&e"
 
_output_shapes
:
��:!f

_output_shapes	
:�:g

_output_shapes
: 
�[
�
C__inference_model_5_layer_call_and_return_conditional_losses_384043
input_169
tubelet_embedding_30_383964: )
tubelet_embedding_30_383966: 9
tubelet_embedding_30_383968: @)
tubelet_embedding_30_383970:@:
tubelet_embedding_30_383972:@�*
tubelet_embedding_30_383974:	�9
tubelet_embedding_31_383977: )
tubelet_embedding_31_383979: 9
tubelet_embedding_31_383981: @)
tubelet_embedding_31_383983:@:
tubelet_embedding_31_383985:@�*
tubelet_embedding_31_383987:	� 
positional_encoder_15_383991/
positional_encoder_15_383993:	(�,
layer_normalization_15_383996:	�,
layer_normalization_15_383998:	�5
multi_head_attention_5_384001:��0
multi_head_attention_5_384003:	�5
multi_head_attention_5_384005:��0
multi_head_attention_5_384007:	�5
multi_head_attention_5_384009:��0
multi_head_attention_5_384011:	�5
multi_head_attention_5_384013:��,
multi_head_attention_5_384015:	�,
layer_normalization_16_384020:	�,
layer_normalization_16_384022:	�'
sequential_5_384025:
��"
sequential_5_384027:	�,
layer_normalization_17_384031:	�,
layer_normalization_17_384033:	�"
dense_11_384037:	�
dense_11_384039:
identity�� dense_11/StatefulPartitionedCall�.layer_normalization_15/StatefulPartitionedCall�.layer_normalization_16/StatefulPartitionedCall�.layer_normalization_17/StatefulPartitionedCall�.multi_head_attention_5/StatefulPartitionedCall�-positional_encoder_15/StatefulPartitionedCall�$sequential_5/StatefulPartitionedCall�,tubelet_embedding_30/StatefulPartitionedCall�,tubelet_embedding_31/StatefulPartitionedCall�
/tf.__operators__.getitem_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_35/strided_sliceStridedSliceinput_168tf.__operators__.getitem_35/strided_slice/stack:output:0:tf.__operators__.getitem_35/strided_slice/stack_1:output:0:tf.__operators__.getitem_35/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
/tf.__operators__.getitem_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_34/strided_sliceStridedSliceinput_168tf.__operators__.getitem_34/strided_slice/stack:output:0:tf.__operators__.getitem_34/strided_slice/stack_1:output:0:tf.__operators__.getitem_34/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
,tubelet_embedding_30/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_34/strided_slice:output:0tubelet_embedding_30_383964tubelet_embedding_30_383966tubelet_embedding_30_383968tubelet_embedding_30_383970tubelet_embedding_30_383972tubelet_embedding_30_383974*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_tubelet_embedding_30_layer_call_and_return_conditional_losses_383031�
,tubelet_embedding_31/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_35/strided_slice:output:0tubelet_embedding_31_383977tubelet_embedding_31_383979tubelet_embedding_31_383981tubelet_embedding_31_383983tubelet_embedding_31_383985tubelet_embedding_31_383987*
Tin
	2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_tubelet_embedding_31_layer_call_and_return_conditional_losses_383081�
concatenate_12/PartitionedCallPartitionedCall5tubelet_embedding_30/StatefulPartitionedCall:output:05tubelet_embedding_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *S
fNRL
J__inference_concatenate_12_layer_call_and_return_conditional_losses_383102�
-positional_encoder_15/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0positional_encoder_15_383991positional_encoder_15_383993*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Z
fURS
Q__inference_positional_encoder_15_layer_call_and_return_conditional_losses_383116�
.layer_normalization_15/StatefulPartitionedCallStatefulPartitionedCall6positional_encoder_15/StatefulPartitionedCall:output:0layer_normalization_15_383996layer_normalization_15_383998*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_383144�
.multi_head_attention_5/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_15/StatefulPartitionedCall:output:07layer_normalization_15/StatefulPartitionedCall:output:0multi_head_attention_5_384001multi_head_attention_5_384003multi_head_attention_5_384005multi_head_attention_5_384007multi_head_attention_5_384009multi_head_attention_5_384011multi_head_attention_5_384013multi_head_attention_5_384015*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:���������(�:���������((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_383482�
add_10/PartitionedCallPartitionedCall7multi_head_attention_5/StatefulPartitionedCall:output:06positional_encoder_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_add_10_layer_call_and_return_conditional_losses_383211�
.layer_normalization_16/StatefulPartitionedCallStatefulPartitionedCalladd_10/PartitionedCall:output:0layer_normalization_16_384020layer_normalization_16_384022*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_16_layer_call_and_return_conditional_losses_383235�
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_16/StatefulPartitionedCall:output:0sequential_5_384025sequential_5_384027*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_382933�
add_11/PartitionedCallPartitionedCall-sequential_5/StatefulPartitionedCall:output:0add_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_add_11_layer_call_and_return_conditional_losses_383252�
.layer_normalization_17/StatefulPartitionedCallStatefulPartitionedCalladd_11/PartitionedCall:output:0layer_normalization_17_384031layer_normalization_17_384033*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_17_layer_call_and_return_conditional_losses_383276�
*global_average_pooling1d_5/PartitionedCallPartitionedCall7layer_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *_
fZRX
V__inference_global_average_pooling1d_5_layer_call_and_return_conditional_losses_382977�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_5/PartitionedCall:output:0dense_11_384037dense_11_384039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_383293x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_11/StatefulPartitionedCall/^layer_normalization_15/StatefulPartitionedCall/^layer_normalization_16/StatefulPartitionedCall/^layer_normalization_17/StatefulPartitionedCall/^multi_head_attention_5/StatefulPartitionedCall.^positional_encoder_15/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall-^tubelet_embedding_30/StatefulPartitionedCall-^tubelet_embedding_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2`
.layer_normalization_15/StatefulPartitionedCall.layer_normalization_15/StatefulPartitionedCall2`
.layer_normalization_16/StatefulPartitionedCall.layer_normalization_16/StatefulPartitionedCall2`
.layer_normalization_17/StatefulPartitionedCall.layer_normalization_17/StatefulPartitionedCall2`
.multi_head_attention_5/StatefulPartitionedCall.multi_head_attention_5/StatefulPartitionedCall2^
-positional_encoder_15/StatefulPartitionedCall-positional_encoder_15/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2\
,tubelet_embedding_30/StatefulPartitionedCall,tubelet_embedding_30/StatefulPartitionedCall2\
,tubelet_embedding_31/StatefulPartitionedCall,tubelet_embedding_31/StatefulPartitionedCall:] Y
3
_output_shapes!
:���������

"
_user_specified_name
input_16: 

_output_shapes
:(
�3
�
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_384991	
query	
valueC
+query_einsum_einsum_readvariableop_resource:��4
!query_add_readvariableop_resource:	�A
)key_einsum_einsum_readvariableop_resource:��2
key_add_readvariableop_resource:	�C
+value_einsum_einsum_readvariableop_resource:��4
!value_add_readvariableop_resource:	�N
6attention_output_einsum_einsum_readvariableop_resource:��;
,attention_output_add_readvariableop_resource:	�
identity

identity_1��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde{
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abdew
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde{
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(�J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:���������(��
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������((*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������((Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������((^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������((*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������((�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������((�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������((�
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:���������(�*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������(�*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:���������(�r

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������((�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:���������(�:���������(�: : : : : : : : 2J
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
:���������(�

_user_specified_namequery:SO
,
_output_shapes
:���������(�

_user_specified_namevalue
�
�
(__inference_model_5_layer_call_fn_383367
input_16%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�'
	unknown_5: 
	unknown_6: '
	unknown_7: @
	unknown_8:@(
	unknown_9:@�

unknown_10:	�

unknown_11

unknown_12:	(�

unknown_13:	�

unknown_14:	�"

unknown_15:��

unknown_16:	�"

unknown_17:��

unknown_18:	�"

unknown_19:��

unknown_20:	�"

unknown_21:��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:���������*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_383300o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
3
_output_shapes!
:���������

"
_user_specified_name
input_16: 

_output_shapes
:(
�
M
1__inference_max_pooling3d_63_layer_call_fn_385246

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_max_pooling3d_63_layer_call_and_return_conditional_losses_382829�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
(__inference_model_5_layer_call_fn_383863
input_16%
unknown: 
	unknown_0: '
	unknown_1: @
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�'
	unknown_5: 
	unknown_6: '
	unknown_7: @
	unknown_8:@(
	unknown_9:@�

unknown_10:	�

unknown_11

unknown_12:	(�

unknown_13:	�

unknown_14:	�"

unknown_15:��

unknown_16:	�"

unknown_17:��

unknown_18:	�"

unknown_19:��

unknown_20:	�"

unknown_21:��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:���������*A
_read_only_resource_inputs#
!	
 *2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_383727o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
3
_output_shapes!
:���������

"
_user_specified_name
input_16: 

_output_shapes
:(
��
�%
!__inference__wrapped_model_382772
input_16c
Emodel_5_tubelet_embedding_30_conv3d_90_conv3d_readvariableop_resource: T
Fmodel_5_tubelet_embedding_30_conv3d_90_biasadd_readvariableop_resource: c
Emodel_5_tubelet_embedding_30_conv3d_91_conv3d_readvariableop_resource: @T
Fmodel_5_tubelet_embedding_30_conv3d_91_biasadd_readvariableop_resource:@d
Emodel_5_tubelet_embedding_30_conv3d_92_conv3d_readvariableop_resource:@�U
Fmodel_5_tubelet_embedding_30_conv3d_92_biasadd_readvariableop_resource:	�c
Emodel_5_tubelet_embedding_31_conv3d_93_conv3d_readvariableop_resource: T
Fmodel_5_tubelet_embedding_31_conv3d_93_biasadd_readvariableop_resource: c
Emodel_5_tubelet_embedding_31_conv3d_94_conv3d_readvariableop_resource: @T
Fmodel_5_tubelet_embedding_31_conv3d_94_biasadd_readvariableop_resource:@d
Emodel_5_tubelet_embedding_31_conv3d_95_conv3d_readvariableop_resource:@�U
Fmodel_5_tubelet_embedding_31_conv3d_95_biasadd_readvariableop_resource:	�(
$model_5_positional_encoder_15_382635R
?model_5_positional_encoder_15_embedding_embedding_lookup_382637:	(�S
Dmodel_5_layer_normalization_15_batchnorm_mul_readvariableop_resource:	�O
@model_5_layer_normalization_15_batchnorm_readvariableop_resource:	�b
Jmodel_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource:��S
@model_5_multi_head_attention_5_query_add_readvariableop_resource:	�`
Hmodel_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource:��Q
>model_5_multi_head_attention_5_key_add_readvariableop_resource:	�b
Jmodel_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource:��S
@model_5_multi_head_attention_5_value_add_readvariableop_resource:	�m
Umodel_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource:��Z
Kmodel_5_multi_head_attention_5_attention_output_add_readvariableop_resource:	�S
Dmodel_5_layer_normalization_16_batchnorm_mul_readvariableop_resource:	�O
@model_5_layer_normalization_16_batchnorm_readvariableop_resource:	�S
?model_5_sequential_5_dense_10_tensordot_readvariableop_resource:
��L
=model_5_sequential_5_dense_10_biasadd_readvariableop_resource:	�S
Dmodel_5_layer_normalization_17_batchnorm_mul_readvariableop_resource:	�O
@model_5_layer_normalization_17_batchnorm_readvariableop_resource:	�B
/model_5_dense_11_matmul_readvariableop_resource:	�>
0model_5_dense_11_biasadd_readvariableop_resource:
identity��'model_5/dense_11/BiasAdd/ReadVariableOp�&model_5/dense_11/MatMul/ReadVariableOp�7model_5/layer_normalization_15/batchnorm/ReadVariableOp�;model_5/layer_normalization_15/batchnorm/mul/ReadVariableOp�7model_5/layer_normalization_16/batchnorm/ReadVariableOp�;model_5/layer_normalization_16/batchnorm/mul/ReadVariableOp�7model_5/layer_normalization_17/batchnorm/ReadVariableOp�;model_5/layer_normalization_17/batchnorm/mul/ReadVariableOp�Bmodel_5/multi_head_attention_5/attention_output/add/ReadVariableOp�Lmodel_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�5model_5/multi_head_attention_5/key/add/ReadVariableOp�?model_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�7model_5/multi_head_attention_5/query/add/ReadVariableOp�Amodel_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�7model_5/multi_head_attention_5/value/add/ReadVariableOp�Amodel_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�8model_5/positional_encoder_15/embedding/embedding_lookup�4model_5/sequential_5/dense_10/BiasAdd/ReadVariableOp�6model_5/sequential_5/dense_10/Tensordot/ReadVariableOp�=model_5/tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp�<model_5/tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp�=model_5/tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp�<model_5/tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp�=model_5/tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp�<model_5/tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp�=model_5/tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp�<model_5/tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp�=model_5/tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp�<model_5/tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp�=model_5/tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp�<model_5/tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp�
7model_5/tf.__operators__.getitem_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   �
9model_5/tf.__operators__.getitem_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    �
9model_5/tf.__operators__.getitem_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
1model_5/tf.__operators__.getitem_35/strided_sliceStridedSliceinput_16@model_5/tf.__operators__.getitem_35/strided_slice/stack:output:0Bmodel_5/tf.__operators__.getitem_35/strided_slice/stack_1:output:0Bmodel_5/tf.__operators__.getitem_35/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
7model_5/tf.__operators__.getitem_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    �
9model_5/tf.__operators__.getitem_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   �
9model_5/tf.__operators__.getitem_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
1model_5/tf.__operators__.getitem_34/strided_sliceStridedSliceinput_16@model_5/tf.__operators__.getitem_34/strided_slice/stack:output:0Bmodel_5/tf.__operators__.getitem_34/strided_slice/stack_1:output:0Bmodel_5/tf.__operators__.getitem_34/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
<model_5/tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOpReadVariableOpEmodel_5_tubelet_embedding_30_conv3d_90_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
-model_5/tubelet_embedding_30/conv3d_90/Conv3DConv3D:model_5/tf.__operators__.getitem_34/strided_slice:output:0Dmodel_5/tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
=model_5/tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOpReadVariableOpFmodel_5_tubelet_embedding_30_conv3d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
.model_5/tubelet_embedding_30/conv3d_90/BiasAddBiasAdd6model_5/tubelet_embedding_30/conv3d_90/Conv3D:output:0Emodel_5/tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 �
+model_5/tubelet_embedding_30/conv3d_90/ReluRelu7model_5/tubelet_embedding_30/conv3d_90/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
7model_5/tubelet_embedding_30/max_pooling3d_60/MaxPool3D	MaxPool3D9model_5/tubelet_embedding_30/conv3d_90/Relu:activations:0*
T0*3
_output_shapes!
:���������
 *
ksize	
*
paddingVALID*
strides	
�
<model_5/tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOpReadVariableOpEmodel_5_tubelet_embedding_30_conv3d_91_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
-model_5/tubelet_embedding_30/conv3d_91/Conv3DConv3D@model_5/tubelet_embedding_30/max_pooling3d_60/MaxPool3D:output:0Dmodel_5/tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@*
paddingSAME*
strides	
�
=model_5/tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOpReadVariableOpFmodel_5_tubelet_embedding_30_conv3d_91_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
.model_5/tubelet_embedding_30/conv3d_91/BiasAddBiasAdd6model_5/tubelet_embedding_30/conv3d_91/Conv3D:output:0Emodel_5/tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@�
+model_5/tubelet_embedding_30/conv3d_91/ReluRelu7model_5/tubelet_embedding_30/conv3d_91/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
@�
7model_5/tubelet_embedding_30/max_pooling3d_61/MaxPool3D	MaxPool3D9model_5/tubelet_embedding_30/conv3d_91/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
<model_5/tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOpReadVariableOpEmodel_5_tubelet_embedding_30_conv3d_92_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
-model_5/tubelet_embedding_30/conv3d_92/Conv3DConv3D@model_5/tubelet_embedding_30/max_pooling3d_61/MaxPool3D:output:0Dmodel_5/tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
=model_5/tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOpReadVariableOpFmodel_5_tubelet_embedding_30_conv3d_92_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.model_5/tubelet_embedding_30/conv3d_92/BiasAddBiasAdd6model_5/tubelet_embedding_30/conv3d_92/Conv3D:output:0Emodel_5/tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
;model_5/tubelet_embedding_30/average_pooling3d_30/AvgPool3D	AvgPool3D7model_5/tubelet_embedding_30/conv3d_92/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
�*
ksize	
*
paddingVALID*
strides	
�
-model_5/tubelet_embedding_30/reshape_30/ShapeShapeDmodel_5/tubelet_embedding_30/average_pooling3d_30/AvgPool3D:output:0*
T0*
_output_shapes
:�
;model_5/tubelet_embedding_30/reshape_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=model_5/tubelet_embedding_30/reshape_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=model_5/tubelet_embedding_30/reshape_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5model_5/tubelet_embedding_30/reshape_30/strided_sliceStridedSlice6model_5/tubelet_embedding_30/reshape_30/Shape:output:0Dmodel_5/tubelet_embedding_30/reshape_30/strided_slice/stack:output:0Fmodel_5/tubelet_embedding_30/reshape_30/strided_slice/stack_1:output:0Fmodel_5/tubelet_embedding_30/reshape_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7model_5/tubelet_embedding_30/reshape_30/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������z
7model_5/tubelet_embedding_30/reshape_30/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
5model_5/tubelet_embedding_30/reshape_30/Reshape/shapePack>model_5/tubelet_embedding_30/reshape_30/strided_slice:output:0@model_5/tubelet_embedding_30/reshape_30/Reshape/shape/1:output:0@model_5/tubelet_embedding_30/reshape_30/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
/model_5/tubelet_embedding_30/reshape_30/ReshapeReshapeDmodel_5/tubelet_embedding_30/average_pooling3d_30/AvgPool3D:output:0>model_5/tubelet_embedding_30/reshape_30/Reshape/shape:output:0*
T0*,
_output_shapes
:���������(��
<model_5/tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOpReadVariableOpEmodel_5_tubelet_embedding_31_conv3d_93_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
-model_5/tubelet_embedding_31/conv3d_93/Conv3DConv3D:model_5/tf.__operators__.getitem_35/strided_slice:output:0Dmodel_5/tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
=model_5/tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOpReadVariableOpFmodel_5_tubelet_embedding_31_conv3d_93_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
.model_5/tubelet_embedding_31/conv3d_93/BiasAddBiasAdd6model_5/tubelet_embedding_31/conv3d_93/Conv3D:output:0Emodel_5/tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 �
+model_5/tubelet_embedding_31/conv3d_93/ReluRelu7model_5/tubelet_embedding_31/conv3d_93/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
7model_5/tubelet_embedding_31/max_pooling3d_62/MaxPool3D	MaxPool3D9model_5/tubelet_embedding_31/conv3d_93/Relu:activations:0*
T0*3
_output_shapes!
:���������
 *
ksize	
*
paddingVALID*
strides	
�
<model_5/tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOpReadVariableOpEmodel_5_tubelet_embedding_31_conv3d_94_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
-model_5/tubelet_embedding_31/conv3d_94/Conv3DConv3D@model_5/tubelet_embedding_31/max_pooling3d_62/MaxPool3D:output:0Dmodel_5/tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@*
paddingSAME*
strides	
�
=model_5/tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOpReadVariableOpFmodel_5_tubelet_embedding_31_conv3d_94_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
.model_5/tubelet_embedding_31/conv3d_94/BiasAddBiasAdd6model_5/tubelet_embedding_31/conv3d_94/Conv3D:output:0Emodel_5/tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@�
+model_5/tubelet_embedding_31/conv3d_94/ReluRelu7model_5/tubelet_embedding_31/conv3d_94/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
@�
7model_5/tubelet_embedding_31/max_pooling3d_63/MaxPool3D	MaxPool3D9model_5/tubelet_embedding_31/conv3d_94/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
<model_5/tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOpReadVariableOpEmodel_5_tubelet_embedding_31_conv3d_95_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
-model_5/tubelet_embedding_31/conv3d_95/Conv3DConv3D@model_5/tubelet_embedding_31/max_pooling3d_63/MaxPool3D:output:0Dmodel_5/tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
=model_5/tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOpReadVariableOpFmodel_5_tubelet_embedding_31_conv3d_95_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.model_5/tubelet_embedding_31/conv3d_95/BiasAddBiasAdd6model_5/tubelet_embedding_31/conv3d_95/Conv3D:output:0Emodel_5/tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
;model_5/tubelet_embedding_31/average_pooling3d_31/AvgPool3D	AvgPool3D7model_5/tubelet_embedding_31/conv3d_95/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
�*
ksize	
*
paddingVALID*
strides	
�
-model_5/tubelet_embedding_31/reshape_31/ShapeShapeDmodel_5/tubelet_embedding_31/average_pooling3d_31/AvgPool3D:output:0*
T0*
_output_shapes
:�
;model_5/tubelet_embedding_31/reshape_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=model_5/tubelet_embedding_31/reshape_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=model_5/tubelet_embedding_31/reshape_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5model_5/tubelet_embedding_31/reshape_31/strided_sliceStridedSlice6model_5/tubelet_embedding_31/reshape_31/Shape:output:0Dmodel_5/tubelet_embedding_31/reshape_31/strided_slice/stack:output:0Fmodel_5/tubelet_embedding_31/reshape_31/strided_slice/stack_1:output:0Fmodel_5/tubelet_embedding_31/reshape_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
7model_5/tubelet_embedding_31/reshape_31/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������z
7model_5/tubelet_embedding_31/reshape_31/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
5model_5/tubelet_embedding_31/reshape_31/Reshape/shapePack>model_5/tubelet_embedding_31/reshape_31/strided_slice:output:0@model_5/tubelet_embedding_31/reshape_31/Reshape/shape/1:output:0@model_5/tubelet_embedding_31/reshape_31/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
/model_5/tubelet_embedding_31/reshape_31/ReshapeReshapeDmodel_5/tubelet_embedding_31/average_pooling3d_31/AvgPool3D:output:0>model_5/tubelet_embedding_31/reshape_31/Reshape/shape:output:0*
T0*,
_output_shapes
:���������(�d
"model_5/concatenate_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_5/concatenate_12/concatConcatV28model_5/tubelet_embedding_30/reshape_30/Reshape:output:08model_5/tubelet_embedding_31/reshape_31/Reshape:output:0+model_5/concatenate_12/concat/axis:output:0*
N*
T0*,
_output_shapes
:���������(��
8model_5/positional_encoder_15/embedding/embedding_lookupResourceGather?model_5_positional_encoder_15_embedding_embedding_lookup_382637$model_5_positional_encoder_15_382635*
Tindices0*R
_classH
FDloc:@model_5/positional_encoder_15/embedding/embedding_lookup/382637*
_output_shapes
:	(�*
dtype0�
Amodel_5/positional_encoder_15/embedding/embedding_lookup/IdentityIdentityAmodel_5/positional_encoder_15/embedding/embedding_lookup:output:0*
T0*R
_classH
FDloc:@model_5/positional_encoder_15/embedding/embedding_lookup/382637*
_output_shapes
:	(��
Cmodel_5/positional_encoder_15/embedding/embedding_lookup/Identity_1IdentityJmodel_5/positional_encoder_15/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	(��
!model_5/positional_encoder_15/addAddV2&model_5/concatenate_12/concat:output:0Lmodel_5/positional_encoder_15/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:���������(��
=model_5/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
+model_5/layer_normalization_15/moments/meanMean%model_5/positional_encoder_15/add:z:0Fmodel_5/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(�
3model_5/layer_normalization_15/moments/StopGradientStopGradient4model_5/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:���������(�
8model_5/layer_normalization_15/moments/SquaredDifferenceSquaredDifference%model_5/positional_encoder_15/add:z:0<model_5/layer_normalization_15/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������(��
Amodel_5/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
/model_5/layer_normalization_15/moments/varianceMean<model_5/layer_normalization_15/moments/SquaredDifference:z:0Jmodel_5/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(s
.model_5/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
,model_5/layer_normalization_15/batchnorm/addAddV28model_5/layer_normalization_15/moments/variance:output:07model_5/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(�
.model_5/layer_normalization_15/batchnorm/RsqrtRsqrt0model_5/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:���������(�
;model_5/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,model_5/layer_normalization_15/batchnorm/mulMul2model_5/layer_normalization_15/batchnorm/Rsqrt:y:0Cmodel_5/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��
.model_5/layer_normalization_15/batchnorm/mul_1Mul%model_5/positional_encoder_15/add:z:00model_5/layer_normalization_15/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
.model_5/layer_normalization_15/batchnorm/mul_2Mul4model_5/layer_normalization_15/moments/mean:output:00model_5/layer_normalization_15/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
7model_5/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp@model_5_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,model_5/layer_normalization_15/batchnorm/subSub?model_5/layer_normalization_15/batchnorm/ReadVariableOp:value:02model_5/layer_normalization_15/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(��
.model_5/layer_normalization_15/batchnorm/add_1AddV22model_5/layer_normalization_15/batchnorm/mul_1:z:00model_5/layer_normalization_15/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(��
Amodel_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpJmodel_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
2model_5/multi_head_attention_5/query/einsum/EinsumEinsum2model_5/layer_normalization_15/batchnorm/add_1:z:0Imodel_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde�
7model_5/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp@model_5_multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
(model_5/multi_head_attention_5/query/addAddV2;model_5/multi_head_attention_5/query/einsum/Einsum:output:0?model_5/multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
?model_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
0model_5/multi_head_attention_5/key/einsum/EinsumEinsum2model_5/layer_normalization_15/batchnorm/add_1:z:0Gmodel_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde�
5model_5/multi_head_attention_5/key/add/ReadVariableOpReadVariableOp>model_5_multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
&model_5/multi_head_attention_5/key/addAddV29model_5/multi_head_attention_5/key/einsum/Einsum:output:0=model_5/multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
Amodel_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpJmodel_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
2model_5/multi_head_attention_5/value/einsum/EinsumEinsum2model_5/layer_normalization_15/batchnorm/add_1:z:0Imodel_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde�
7model_5/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp@model_5_multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
(model_5/multi_head_attention_5/value/addAddV2;model_5/multi_head_attention_5/value/einsum/Einsum:output:0?model_5/multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(�i
$model_5/multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
"model_5/multi_head_attention_5/MulMul,model_5/multi_head_attention_5/query/add:z:0-model_5/multi_head_attention_5/Mul/y:output:0*
T0*0
_output_shapes
:���������(��
,model_5/multi_head_attention_5/einsum/EinsumEinsum*model_5/multi_head_attention_5/key/add:z:0&model_5/multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:���������((*
equationaecd,abcd->acbe�
.model_5/multi_head_attention_5/softmax/SoftmaxSoftmax5model_5/multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������((�
/model_5/multi_head_attention_5/dropout/IdentityIdentity8model_5/multi_head_attention_5/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������((�
.model_5/multi_head_attention_5/einsum_1/EinsumEinsum8model_5/multi_head_attention_5/dropout/Identity:output:0,model_5/multi_head_attention_5/value/add:z:0*
N*
T0*0
_output_shapes
:���������(�*
equationacbe,aecd->abcd�
Lmodel_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpUmodel_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
=model_5/multi_head_attention_5/attention_output/einsum/EinsumEinsum7model_5/multi_head_attention_5/einsum_1/Einsum:output:0Tmodel_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������(�*
equationabcd,cde->abe�
Bmodel_5/multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpKmodel_5_multi_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
3model_5/multi_head_attention_5/attention_output/addAddV2Fmodel_5/multi_head_attention_5/attention_output/einsum/Einsum:output:0Jmodel_5/multi_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��
model_5/add_10/addAddV27model_5/multi_head_attention_5/attention_output/add:z:0%model_5/positional_encoder_15/add:z:0*
T0*,
_output_shapes
:���������(��
=model_5/layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
+model_5/layer_normalization_16/moments/meanMeanmodel_5/add_10/add:z:0Fmodel_5/layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(�
3model_5/layer_normalization_16/moments/StopGradientStopGradient4model_5/layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:���������(�
8model_5/layer_normalization_16/moments/SquaredDifferenceSquaredDifferencemodel_5/add_10/add:z:0<model_5/layer_normalization_16/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������(��
Amodel_5/layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
/model_5/layer_normalization_16/moments/varianceMean<model_5/layer_normalization_16/moments/SquaredDifference:z:0Jmodel_5/layer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(s
.model_5/layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
,model_5/layer_normalization_16/batchnorm/addAddV28model_5/layer_normalization_16/moments/variance:output:07model_5/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(�
.model_5/layer_normalization_16/batchnorm/RsqrtRsqrt0model_5/layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:���������(�
;model_5/layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,model_5/layer_normalization_16/batchnorm/mulMul2model_5/layer_normalization_16/batchnorm/Rsqrt:y:0Cmodel_5/layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��
.model_5/layer_normalization_16/batchnorm/mul_1Mulmodel_5/add_10/add:z:00model_5/layer_normalization_16/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
.model_5/layer_normalization_16/batchnorm/mul_2Mul4model_5/layer_normalization_16/moments/mean:output:00model_5/layer_normalization_16/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
7model_5/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp@model_5_layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,model_5/layer_normalization_16/batchnorm/subSub?model_5/layer_normalization_16/batchnorm/ReadVariableOp:value:02model_5/layer_normalization_16/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(��
.model_5/layer_normalization_16/batchnorm/add_1AddV22model_5/layer_normalization_16/batchnorm/mul_1:z:00model_5/layer_normalization_16/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(��
6model_5/sequential_5/dense_10/Tensordot/ReadVariableOpReadVariableOp?model_5_sequential_5_dense_10_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0v
,model_5/sequential_5/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,model_5/sequential_5/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
-model_5/sequential_5/dense_10/Tensordot/ShapeShape2model_5/layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
5model_5/sequential_5/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_5/sequential_5/dense_10/Tensordot/GatherV2GatherV26model_5/sequential_5/dense_10/Tensordot/Shape:output:05model_5/sequential_5/dense_10/Tensordot/free:output:0>model_5/sequential_5/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7model_5/sequential_5/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
2model_5/sequential_5/dense_10/Tensordot/GatherV2_1GatherV26model_5/sequential_5/dense_10/Tensordot/Shape:output:05model_5/sequential_5/dense_10/Tensordot/axes:output:0@model_5/sequential_5/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-model_5/sequential_5/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
,model_5/sequential_5/dense_10/Tensordot/ProdProd9model_5/sequential_5/dense_10/Tensordot/GatherV2:output:06model_5/sequential_5/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/model_5/sequential_5/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
.model_5/sequential_5/dense_10/Tensordot/Prod_1Prod;model_5/sequential_5/dense_10/Tensordot/GatherV2_1:output:08model_5/sequential_5/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3model_5/sequential_5/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.model_5/sequential_5/dense_10/Tensordot/concatConcatV25model_5/sequential_5/dense_10/Tensordot/free:output:05model_5/sequential_5/dense_10/Tensordot/axes:output:0<model_5/sequential_5/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
-model_5/sequential_5/dense_10/Tensordot/stackPack5model_5/sequential_5/dense_10/Tensordot/Prod:output:07model_5/sequential_5/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
1model_5/sequential_5/dense_10/Tensordot/transpose	Transpose2model_5/layer_normalization_16/batchnorm/add_1:z:07model_5/sequential_5/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������(��
/model_5/sequential_5/dense_10/Tensordot/ReshapeReshape5model_5/sequential_5/dense_10/Tensordot/transpose:y:06model_5/sequential_5/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
.model_5/sequential_5/dense_10/Tensordot/MatMulMatMul8model_5/sequential_5/dense_10/Tensordot/Reshape:output:0>model_5/sequential_5/dense_10/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
/model_5/sequential_5/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�w
5model_5/sequential_5/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
0model_5/sequential_5/dense_10/Tensordot/concat_1ConcatV29model_5/sequential_5/dense_10/Tensordot/GatherV2:output:08model_5/sequential_5/dense_10/Tensordot/Const_2:output:0>model_5/sequential_5/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
'model_5/sequential_5/dense_10/TensordotReshape8model_5/sequential_5/dense_10/Tensordot/MatMul:product:09model_5/sequential_5/dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������(��
4model_5/sequential_5/dense_10/BiasAdd/ReadVariableOpReadVariableOp=model_5_sequential_5_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_5/sequential_5/dense_10/BiasAddBiasAdd0model_5/sequential_5/dense_10/Tensordot:output:0<model_5/sequential_5/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�m
(model_5/sequential_5/dense_10/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
&model_5/sequential_5/dense_10/Gelu/mulMul1model_5/sequential_5/dense_10/Gelu/mul/x:output:0.model_5/sequential_5/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:���������(�n
)model_5/sequential_5/dense_10/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
*model_5/sequential_5/dense_10/Gelu/truedivRealDiv.model_5/sequential_5/dense_10/BiasAdd:output:02model_5/sequential_5/dense_10/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������(��
&model_5/sequential_5/dense_10/Gelu/ErfErf.model_5/sequential_5/dense_10/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������(�m
(model_5/sequential_5/dense_10/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
&model_5/sequential_5/dense_10/Gelu/addAddV21model_5/sequential_5/dense_10/Gelu/add/x:output:0*model_5/sequential_5/dense_10/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������(��
(model_5/sequential_5/dense_10/Gelu/mul_1Mul*model_5/sequential_5/dense_10/Gelu/mul:z:0*model_5/sequential_5/dense_10/Gelu/add:z:0*
T0*,
_output_shapes
:���������(��
model_5/add_11/addAddV2,model_5/sequential_5/dense_10/Gelu/mul_1:z:0model_5/add_10/add:z:0*
T0*,
_output_shapes
:���������(��
=model_5/layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
+model_5/layer_normalization_17/moments/meanMeanmodel_5/add_11/add:z:0Fmodel_5/layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(�
3model_5/layer_normalization_17/moments/StopGradientStopGradient4model_5/layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:���������(�
8model_5/layer_normalization_17/moments/SquaredDifferenceSquaredDifferencemodel_5/add_11/add:z:0<model_5/layer_normalization_17/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������(��
Amodel_5/layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
/model_5/layer_normalization_17/moments/varianceMean<model_5/layer_normalization_17/moments/SquaredDifference:z:0Jmodel_5/layer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(s
.model_5/layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
,model_5/layer_normalization_17/batchnorm/addAddV28model_5/layer_normalization_17/moments/variance:output:07model_5/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(�
.model_5/layer_normalization_17/batchnorm/RsqrtRsqrt0model_5/layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:���������(�
;model_5/layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,model_5/layer_normalization_17/batchnorm/mulMul2model_5/layer_normalization_17/batchnorm/Rsqrt:y:0Cmodel_5/layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��
.model_5/layer_normalization_17/batchnorm/mul_1Mulmodel_5/add_11/add:z:00model_5/layer_normalization_17/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
.model_5/layer_normalization_17/batchnorm/mul_2Mul4model_5/layer_normalization_17/moments/mean:output:00model_5/layer_normalization_17/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
7model_5/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp@model_5_layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,model_5/layer_normalization_17/batchnorm/subSub?model_5/layer_normalization_17/batchnorm/ReadVariableOp:value:02model_5/layer_normalization_17/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(��
.model_5/layer_normalization_17/batchnorm/add_1AddV22model_5/layer_normalization_17/batchnorm/mul_1:z:00model_5/layer_normalization_17/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(�{
9model_5/global_average_pooling1d_5/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'model_5/global_average_pooling1d_5/MeanMean2model_5/layer_normalization_17/batchnorm/add_1:z:0Bmodel_5/global_average_pooling1d_5/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:�����������
&model_5/dense_11/MatMul/ReadVariableOpReadVariableOp/model_5_dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_5/dense_11/MatMulMatMul0model_5/global_average_pooling1d_5/Mean:output:0.model_5/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_5/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_5/dense_11/BiasAddBiasAdd!model_5/dense_11/MatMul:product:0/model_5/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
IdentityIdentity!model_5/dense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^model_5/dense_11/BiasAdd/ReadVariableOp'^model_5/dense_11/MatMul/ReadVariableOp8^model_5/layer_normalization_15/batchnorm/ReadVariableOp<^model_5/layer_normalization_15/batchnorm/mul/ReadVariableOp8^model_5/layer_normalization_16/batchnorm/ReadVariableOp<^model_5/layer_normalization_16/batchnorm/mul/ReadVariableOp8^model_5/layer_normalization_17/batchnorm/ReadVariableOp<^model_5/layer_normalization_17/batchnorm/mul/ReadVariableOpC^model_5/multi_head_attention_5/attention_output/add/ReadVariableOpM^model_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp6^model_5/multi_head_attention_5/key/add/ReadVariableOp@^model_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp8^model_5/multi_head_attention_5/query/add/ReadVariableOpB^model_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp8^model_5/multi_head_attention_5/value/add/ReadVariableOpB^model_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9^model_5/positional_encoder_15/embedding/embedding_lookup5^model_5/sequential_5/dense_10/BiasAdd/ReadVariableOp7^model_5/sequential_5/dense_10/Tensordot/ReadVariableOp>^model_5/tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp=^model_5/tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp>^model_5/tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp=^model_5/tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp>^model_5/tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp=^model_5/tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp>^model_5/tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp=^model_5/tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp>^model_5/tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp=^model_5/tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp>^model_5/tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp=^model_5/tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2R
'model_5/dense_11/BiasAdd/ReadVariableOp'model_5/dense_11/BiasAdd/ReadVariableOp2P
&model_5/dense_11/MatMul/ReadVariableOp&model_5/dense_11/MatMul/ReadVariableOp2r
7model_5/layer_normalization_15/batchnorm/ReadVariableOp7model_5/layer_normalization_15/batchnorm/ReadVariableOp2z
;model_5/layer_normalization_15/batchnorm/mul/ReadVariableOp;model_5/layer_normalization_15/batchnorm/mul/ReadVariableOp2r
7model_5/layer_normalization_16/batchnorm/ReadVariableOp7model_5/layer_normalization_16/batchnorm/ReadVariableOp2z
;model_5/layer_normalization_16/batchnorm/mul/ReadVariableOp;model_5/layer_normalization_16/batchnorm/mul/ReadVariableOp2r
7model_5/layer_normalization_17/batchnorm/ReadVariableOp7model_5/layer_normalization_17/batchnorm/ReadVariableOp2z
;model_5/layer_normalization_17/batchnorm/mul/ReadVariableOp;model_5/layer_normalization_17/batchnorm/mul/ReadVariableOp2�
Bmodel_5/multi_head_attention_5/attention_output/add/ReadVariableOpBmodel_5/multi_head_attention_5/attention_output/add/ReadVariableOp2�
Lmodel_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpLmodel_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2n
5model_5/multi_head_attention_5/key/add/ReadVariableOp5model_5/multi_head_attention_5/key/add/ReadVariableOp2�
?model_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp?model_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2r
7model_5/multi_head_attention_5/query/add/ReadVariableOp7model_5/multi_head_attention_5/query/add/ReadVariableOp2�
Amodel_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpAmodel_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2r
7model_5/multi_head_attention_5/value/add/ReadVariableOp7model_5/multi_head_attention_5/value/add/ReadVariableOp2�
Amodel_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpAmodel_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2t
8model_5/positional_encoder_15/embedding/embedding_lookup8model_5/positional_encoder_15/embedding/embedding_lookup2l
4model_5/sequential_5/dense_10/BiasAdd/ReadVariableOp4model_5/sequential_5/dense_10/BiasAdd/ReadVariableOp2p
6model_5/sequential_5/dense_10/Tensordot/ReadVariableOp6model_5/sequential_5/dense_10/Tensordot/ReadVariableOp2~
=model_5/tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp=model_5/tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp2|
<model_5/tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp<model_5/tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp2~
=model_5/tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp=model_5/tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp2|
<model_5/tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp<model_5/tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp2~
=model_5/tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp=model_5/tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp2|
<model_5/tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp<model_5/tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp2~
=model_5/tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp=model_5/tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp2|
<model_5/tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp<model_5/tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp2~
=model_5/tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp=model_5/tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp2|
<model_5/tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp<model_5/tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp2~
=model_5/tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp=model_5/tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp2|
<model_5/tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp<model_5/tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp:] Y
3
_output_shapes!
:���������

"
_user_specified_name
input_16: 

_output_shapes
:(
�
t
J__inference_concatenate_12_layer_call_and_return_conditional_losses_383102

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
:���������(�\
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:���������(�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������(�:���������(�:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs:TP
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
�
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_384864

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������(�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������(�l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�h
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(�g
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
r
V__inference_global_average_pooling1d_5_layer_call_and_return_conditional_losses_382977

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
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling3d_62_layer_call_and_return_conditional_losses_382817

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
n
B__inference_add_10_layer_call_and_return_conditional_losses_385003
inputs_0
inputs_1
identityW
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:���������(�T
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:���������(�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������(�:���������(�:V R
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/1
�3
�
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_383482	
query	
valueC
+query_einsum_einsum_readvariableop_resource:��4
!query_add_readvariableop_resource:	�A
)key_einsum_einsum_readvariableop_resource:��2
key_add_readvariableop_resource:	�C
+value_einsum_einsum_readvariableop_resource:��4
!value_add_readvariableop_resource:	�N
6attention_output_einsum_einsum_readvariableop_resource:��;
,attention_output_add_readvariableop_resource:	�
identity

identity_1��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde{
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abdew
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde{
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(�J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:���������(��
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������((*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������((Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������((^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������((*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������((�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������((�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������((�
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:���������(�*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������(�*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:���������(�r

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������((�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:���������(�:���������(�: : : : : : : : 2J
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
:���������(�

_user_specified_namequery:SO
,
_output_shapes
:���������(�

_user_specified_namevalue
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_382933

inputs#
dense_10_382927:
��
dense_10_382929:	�
identity�� dense_10/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_382927dense_10_382929*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_382889}
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�i
NoOpNoOp!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
l
P__inference_average_pooling3d_31_layer_call_and_return_conditional_losses_382841

inputs
identity�
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling3d_63_layer_call_and_return_conditional_losses_382829

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_dense_10_layer_call_fn_385270

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_382889t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
�
7__inference_layer_normalization_16_layer_call_fn_385012

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_16_layer_call_and_return_conditional_losses_383235t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
�
7__inference_layer_normalization_15_layer_call_fn_384842

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_383144t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
h
L__inference_max_pooling3d_63_layer_call_and_return_conditional_losses_385251

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
l
P__inference_average_pooling3d_30_layer_call_and_return_conditional_losses_382805

inputs
identity�
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�+
�
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_383186	
query	
valueC
+query_einsum_einsum_readvariableop_resource:��4
!query_add_readvariableop_resource:	�A
)key_einsum_einsum_readvariableop_resource:��2
key_add_readvariableop_resource:	�C
+value_einsum_einsum_readvariableop_resource:��4
!value_add_readvariableop_resource:	�N
6attention_output_einsum_einsum_readvariableop_resource:��;
,attention_output_add_readvariableop_resource:	�
identity

identity_1��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde{
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abdew
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde{
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(�J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:���������(��
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������((*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������((q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������((�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:���������(�*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������(�*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:���������(�r

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������((�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:���������(�:���������(�: : : : : : : : 2J
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
:���������(�

_user_specified_namequery:SO
,
_output_shapes
:���������(�

_user_specified_namevalue
�

�
Q__inference_positional_encoder_15_layer_call_and_return_conditional_losses_384833
encoded_tokens
unknown4
!embedding_embedding_lookup_384826:	(�
identity��embedding/embedding_lookup�
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_384826unknown*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/384826*
_output_shapes
:	(�*
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/384826*
_output_shapes
:	(��
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	(��
addAddV2encoded_tokens.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:���������(�[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:���������(�c
NoOpNoOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������(�:(: 28
embedding/embedding_lookupembedding/embedding_lookup:\ X
,
_output_shapes
:���������(�
(
_user_specified_nameencoded_tokens: 

_output_shapes
:(
�-
�
P__inference_tubelet_embedding_30_layer_call_and_return_conditional_losses_383031

videosF
(conv3d_90_conv3d_readvariableop_resource: 7
)conv3d_90_biasadd_readvariableop_resource: F
(conv3d_91_conv3d_readvariableop_resource: @7
)conv3d_91_biasadd_readvariableop_resource:@G
(conv3d_92_conv3d_readvariableop_resource:@�8
)conv3d_92_biasadd_readvariableop_resource:	�
identity�� conv3d_90/BiasAdd/ReadVariableOp�conv3d_90/Conv3D/ReadVariableOp� conv3d_91/BiasAdd/ReadVariableOp�conv3d_91/Conv3D/ReadVariableOp� conv3d_92/BiasAdd/ReadVariableOp�conv3d_92/Conv3D/ReadVariableOp�
conv3d_90/Conv3D/ReadVariableOpReadVariableOp(conv3d_90_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
conv3d_90/Conv3DConv3Dvideos'conv3d_90/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
 conv3d_90/BiasAdd/ReadVariableOpReadVariableOp)conv3d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv3d_90/BiasAddBiasAddconv3d_90/Conv3D:output:0(conv3d_90/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 p
conv3d_90/ReluReluconv3d_90/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
max_pooling3d_60/MaxPool3D	MaxPool3Dconv3d_90/Relu:activations:0*
T0*3
_output_shapes!
:���������
 *
ksize	
*
paddingVALID*
strides	
�
conv3d_91/Conv3D/ReadVariableOpReadVariableOp(conv3d_91_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
conv3d_91/Conv3DConv3D#max_pooling3d_60/MaxPool3D:output:0'conv3d_91/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@*
paddingSAME*
strides	
�
 conv3d_91/BiasAdd/ReadVariableOpReadVariableOp)conv3d_91_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d_91/BiasAddBiasAddconv3d_91/Conv3D:output:0(conv3d_91/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@p
conv3d_91/ReluReluconv3d_91/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
@�
max_pooling3d_61/MaxPool3D	MaxPool3Dconv3d_91/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
conv3d_92/Conv3D/ReadVariableOpReadVariableOp(conv3d_92_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
conv3d_92/Conv3DConv3D#max_pooling3d_61/MaxPool3D:output:0'conv3d_92/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
 conv3d_92/BiasAdd/ReadVariableOpReadVariableOp)conv3d_92_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_92/BiasAddBiasAddconv3d_92/Conv3D:output:0(conv3d_92/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
average_pooling3d_30/AvgPool3D	AvgPool3Dconv3d_92/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
�*
ksize	
*
paddingVALID*
strides	
g
reshape_30/ShapeShape'average_pooling3d_30/AvgPool3D:output:0*
T0*
_output_shapes
:h
reshape_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_30/strided_sliceStridedSlicereshape_30/Shape:output:0'reshape_30/strided_slice/stack:output:0)reshape_30/strided_slice/stack_1:output:0)reshape_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
reshape_30/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������]
reshape_30/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_30/Reshape/shapePack!reshape_30/strided_slice:output:0#reshape_30/Reshape/shape/1:output:0#reshape_30/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
reshape_30/ReshapeReshape'average_pooling3d_30/AvgPool3D:output:0!reshape_30/Reshape/shape:output:0*
T0*,
_output_shapes
:���������(�o
IdentityIdentityreshape_30/Reshape:output:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp!^conv3d_90/BiasAdd/ReadVariableOp ^conv3d_90/Conv3D/ReadVariableOp!^conv3d_91/BiasAdd/ReadVariableOp ^conv3d_91/Conv3D/ReadVariableOp!^conv3d_92/BiasAdd/ReadVariableOp ^conv3d_92/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������
: : : : : : 2D
 conv3d_90/BiasAdd/ReadVariableOp conv3d_90/BiasAdd/ReadVariableOp2B
conv3d_90/Conv3D/ReadVariableOpconv3d_90/Conv3D/ReadVariableOp2D
 conv3d_91/BiasAdd/ReadVariableOp conv3d_91/BiasAdd/ReadVariableOp2B
conv3d_91/Conv3D/ReadVariableOpconv3d_91/Conv3D/ReadVariableOp2D
 conv3d_92/BiasAdd/ReadVariableOp conv3d_92/BiasAdd/ReadVariableOp2B
conv3d_92/Conv3D/ReadVariableOpconv3d_92/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������

 
_user_specified_namevideos
�
S
'__inference_add_11_layer_call_fn_385134
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_add_11_layer_call_and_return_conditional_losses_383252e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������(�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������(�:���������(�:V R
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:���������(�
"
_user_specified_name
inputs/1
�
�
6__inference_positional_encoder_15_layer_call_fn_384821
encoded_tokens
unknown
	unknown_0:	(�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoded_tokensunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Z
fURS
Q__inference_positional_encoder_15_layer_call_and_return_conditional_losses_383116t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������(�:(: 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:���������(�
(
_user_specified_nameencoded_tokens: 

_output_shapes
:(
�
h
L__inference_max_pooling3d_60_layer_call_and_return_conditional_losses_385211

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
��
�!
C__inference_model_5_layer_call_and_return_conditional_losses_384401

inputs[
=tubelet_embedding_30_conv3d_90_conv3d_readvariableop_resource: L
>tubelet_embedding_30_conv3d_90_biasadd_readvariableop_resource: [
=tubelet_embedding_30_conv3d_91_conv3d_readvariableop_resource: @L
>tubelet_embedding_30_conv3d_91_biasadd_readvariableop_resource:@\
=tubelet_embedding_30_conv3d_92_conv3d_readvariableop_resource:@�M
>tubelet_embedding_30_conv3d_92_biasadd_readvariableop_resource:	�[
=tubelet_embedding_31_conv3d_93_conv3d_readvariableop_resource: L
>tubelet_embedding_31_conv3d_93_biasadd_readvariableop_resource: [
=tubelet_embedding_31_conv3d_94_conv3d_readvariableop_resource: @L
>tubelet_embedding_31_conv3d_94_biasadd_readvariableop_resource:@\
=tubelet_embedding_31_conv3d_95_conv3d_readvariableop_resource:@�M
>tubelet_embedding_31_conv3d_95_biasadd_readvariableop_resource:	� 
positional_encoder_15_384264J
7positional_encoder_15_embedding_embedding_lookup_384266:	(�K
<layer_normalization_15_batchnorm_mul_readvariableop_resource:	�G
8layer_normalization_15_batchnorm_readvariableop_resource:	�Z
Bmulti_head_attention_5_query_einsum_einsum_readvariableop_resource:��K
8multi_head_attention_5_query_add_readvariableop_resource:	�X
@multi_head_attention_5_key_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_5_key_add_readvariableop_resource:	�Z
Bmulti_head_attention_5_value_einsum_einsum_readvariableop_resource:��K
8multi_head_attention_5_value_add_readvariableop_resource:	�e
Mmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource:��R
Cmulti_head_attention_5_attention_output_add_readvariableop_resource:	�K
<layer_normalization_16_batchnorm_mul_readvariableop_resource:	�G
8layer_normalization_16_batchnorm_readvariableop_resource:	�K
7sequential_5_dense_10_tensordot_readvariableop_resource:
��D
5sequential_5_dense_10_biasadd_readvariableop_resource:	�K
<layer_normalization_17_batchnorm_mul_readvariableop_resource:	�G
8layer_normalization_17_batchnorm_readvariableop_resource:	�:
'dense_11_matmul_readvariableop_resource:	�6
(dense_11_biasadd_readvariableop_resource:
identity��dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�/layer_normalization_15/batchnorm/ReadVariableOp�3layer_normalization_15/batchnorm/mul/ReadVariableOp�/layer_normalization_16/batchnorm/ReadVariableOp�3layer_normalization_16/batchnorm/mul/ReadVariableOp�/layer_normalization_17/batchnorm/ReadVariableOp�3layer_normalization_17/batchnorm/mul/ReadVariableOp�:multi_head_attention_5/attention_output/add/ReadVariableOp�Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_5/key/add/ReadVariableOp�7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/query/add/ReadVariableOp�9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/value/add/ReadVariableOp�9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�0positional_encoder_15/embedding/embedding_lookup�,sequential_5/dense_10/BiasAdd/ReadVariableOp�.sequential_5/dense_10/Tensordot/ReadVariableOp�5tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp�4tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp�5tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp�4tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp�5tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp�4tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp�5tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp�4tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp�5tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp�4tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp�5tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp�4tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp�
/tf.__operators__.getitem_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_35/strided_sliceStridedSliceinputs8tf.__operators__.getitem_35/strided_slice/stack:output:0:tf.__operators__.getitem_35/strided_slice/stack_1:output:0:tf.__operators__.getitem_35/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
/tf.__operators__.getitem_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_34/strided_sliceStridedSliceinputs8tf.__operators__.getitem_34/strided_slice/stack:output:0:tf.__operators__.getitem_34/strided_slice/stack_1:output:0:tf.__operators__.getitem_34/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
4tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_30_conv3d_90_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
%tubelet_embedding_30/conv3d_90/Conv3DConv3D2tf.__operators__.getitem_34/strided_slice:output:0<tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
5tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_30_conv3d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
&tubelet_embedding_30/conv3d_90/BiasAddBiasAdd.tubelet_embedding_30/conv3d_90/Conv3D:output:0=tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 �
#tubelet_embedding_30/conv3d_90/ReluRelu/tubelet_embedding_30/conv3d_90/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
/tubelet_embedding_30/max_pooling3d_60/MaxPool3D	MaxPool3D1tubelet_embedding_30/conv3d_90/Relu:activations:0*
T0*3
_output_shapes!
:���������
 *
ksize	
*
paddingVALID*
strides	
�
4tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_30_conv3d_91_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
%tubelet_embedding_30/conv3d_91/Conv3DConv3D8tubelet_embedding_30/max_pooling3d_60/MaxPool3D:output:0<tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@*
paddingSAME*
strides	
�
5tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_30_conv3d_91_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&tubelet_embedding_30/conv3d_91/BiasAddBiasAdd.tubelet_embedding_30/conv3d_91/Conv3D:output:0=tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@�
#tubelet_embedding_30/conv3d_91/ReluRelu/tubelet_embedding_30/conv3d_91/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
@�
/tubelet_embedding_30/max_pooling3d_61/MaxPool3D	MaxPool3D1tubelet_embedding_30/conv3d_91/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
4tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_30_conv3d_92_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
%tubelet_embedding_30/conv3d_92/Conv3DConv3D8tubelet_embedding_30/max_pooling3d_61/MaxPool3D:output:0<tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
5tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_30_conv3d_92_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&tubelet_embedding_30/conv3d_92/BiasAddBiasAdd.tubelet_embedding_30/conv3d_92/Conv3D:output:0=tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
3tubelet_embedding_30/average_pooling3d_30/AvgPool3D	AvgPool3D/tubelet_embedding_30/conv3d_92/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
�*
ksize	
*
paddingVALID*
strides	
�
%tubelet_embedding_30/reshape_30/ShapeShape<tubelet_embedding_30/average_pooling3d_30/AvgPool3D:output:0*
T0*
_output_shapes
:}
3tubelet_embedding_30/reshape_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tubelet_embedding_30/reshape_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tubelet_embedding_30/reshape_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-tubelet_embedding_30/reshape_30/strided_sliceStridedSlice.tubelet_embedding_30/reshape_30/Shape:output:0<tubelet_embedding_30/reshape_30/strided_slice/stack:output:0>tubelet_embedding_30/reshape_30/strided_slice/stack_1:output:0>tubelet_embedding_30/reshape_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/tubelet_embedding_30/reshape_30/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������r
/tubelet_embedding_30/reshape_30/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
-tubelet_embedding_30/reshape_30/Reshape/shapePack6tubelet_embedding_30/reshape_30/strided_slice:output:08tubelet_embedding_30/reshape_30/Reshape/shape/1:output:08tubelet_embedding_30/reshape_30/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
'tubelet_embedding_30/reshape_30/ReshapeReshape<tubelet_embedding_30/average_pooling3d_30/AvgPool3D:output:06tubelet_embedding_30/reshape_30/Reshape/shape:output:0*
T0*,
_output_shapes
:���������(��
4tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_31_conv3d_93_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
%tubelet_embedding_31/conv3d_93/Conv3DConv3D2tf.__operators__.getitem_35/strided_slice:output:0<tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
5tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_31_conv3d_93_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
&tubelet_embedding_31/conv3d_93/BiasAddBiasAdd.tubelet_embedding_31/conv3d_93/Conv3D:output:0=tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 �
#tubelet_embedding_31/conv3d_93/ReluRelu/tubelet_embedding_31/conv3d_93/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
/tubelet_embedding_31/max_pooling3d_62/MaxPool3D	MaxPool3D1tubelet_embedding_31/conv3d_93/Relu:activations:0*
T0*3
_output_shapes!
:���������
 *
ksize	
*
paddingVALID*
strides	
�
4tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_31_conv3d_94_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
%tubelet_embedding_31/conv3d_94/Conv3DConv3D8tubelet_embedding_31/max_pooling3d_62/MaxPool3D:output:0<tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@*
paddingSAME*
strides	
�
5tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_31_conv3d_94_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&tubelet_embedding_31/conv3d_94/BiasAddBiasAdd.tubelet_embedding_31/conv3d_94/Conv3D:output:0=tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@�
#tubelet_embedding_31/conv3d_94/ReluRelu/tubelet_embedding_31/conv3d_94/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
@�
/tubelet_embedding_31/max_pooling3d_63/MaxPool3D	MaxPool3D1tubelet_embedding_31/conv3d_94/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
4tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_31_conv3d_95_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
%tubelet_embedding_31/conv3d_95/Conv3DConv3D8tubelet_embedding_31/max_pooling3d_63/MaxPool3D:output:0<tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
5tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_31_conv3d_95_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&tubelet_embedding_31/conv3d_95/BiasAddBiasAdd.tubelet_embedding_31/conv3d_95/Conv3D:output:0=tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
3tubelet_embedding_31/average_pooling3d_31/AvgPool3D	AvgPool3D/tubelet_embedding_31/conv3d_95/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
�*
ksize	
*
paddingVALID*
strides	
�
%tubelet_embedding_31/reshape_31/ShapeShape<tubelet_embedding_31/average_pooling3d_31/AvgPool3D:output:0*
T0*
_output_shapes
:}
3tubelet_embedding_31/reshape_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tubelet_embedding_31/reshape_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tubelet_embedding_31/reshape_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-tubelet_embedding_31/reshape_31/strided_sliceStridedSlice.tubelet_embedding_31/reshape_31/Shape:output:0<tubelet_embedding_31/reshape_31/strided_slice/stack:output:0>tubelet_embedding_31/reshape_31/strided_slice/stack_1:output:0>tubelet_embedding_31/reshape_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/tubelet_embedding_31/reshape_31/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������r
/tubelet_embedding_31/reshape_31/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
-tubelet_embedding_31/reshape_31/Reshape/shapePack6tubelet_embedding_31/reshape_31/strided_slice:output:08tubelet_embedding_31/reshape_31/Reshape/shape/1:output:08tubelet_embedding_31/reshape_31/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
'tubelet_embedding_31/reshape_31/ReshapeReshape<tubelet_embedding_31/average_pooling3d_31/AvgPool3D:output:06tubelet_embedding_31/reshape_31/Reshape/shape:output:0*
T0*,
_output_shapes
:���������(�\
concatenate_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_12/concatConcatV20tubelet_embedding_30/reshape_30/Reshape:output:00tubelet_embedding_31/reshape_31/Reshape:output:0#concatenate_12/concat/axis:output:0*
N*
T0*,
_output_shapes
:���������(��
0positional_encoder_15/embedding/embedding_lookupResourceGather7positional_encoder_15_embedding_embedding_lookup_384266positional_encoder_15_384264*
Tindices0*J
_class@
><loc:@positional_encoder_15/embedding/embedding_lookup/384266*
_output_shapes
:	(�*
dtype0�
9positional_encoder_15/embedding/embedding_lookup/IdentityIdentity9positional_encoder_15/embedding/embedding_lookup:output:0*
T0*J
_class@
><loc:@positional_encoder_15/embedding/embedding_lookup/384266*
_output_shapes
:	(��
;positional_encoder_15/embedding/embedding_lookup/Identity_1IdentityBpositional_encoder_15/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	(��
positional_encoder_15/addAddV2concatenate_12/concat:output:0Dpositional_encoder_15/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:���������(�
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_15/moments/meanMeanpositional_encoder_15/add:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(�
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:���������(�
0layer_normalization_15/moments/SquaredDifferenceSquaredDifferencepositional_encoder_15/add:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������(��
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(k
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(�
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:���������(�
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_15/batchnorm/mul_1Mulpositional_encoder_15/add:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(��
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
*multi_head_attention_5/query/einsum/EinsumEinsum*layer_normalization_15/batchnorm/add_1:z:0Amulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde�
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention_5/key/einsum/EinsumEinsum*layer_normalization_15/batchnorm/add_1:z:0?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde�
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
*multi_head_attention_5/value/einsum/EinsumEinsum*layer_normalization_15/batchnorm/add_1:z:0Amulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde�
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(�a
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*0
_output_shapes
:���������(��
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:���������((*
equationaecd,abcd->acbe�
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������((�
'multi_head_attention_5/dropout/IdentityIdentity0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������((�
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/Identity:output:0$multi_head_attention_5/value/add:z:0*
N*
T0*0
_output_shapes
:���������(�*
equationacbe,aecd->abcd�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������(�*
equationabcd,cde->abe�
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��

add_10/addAddV2/multi_head_attention_5/attention_output/add:z:0positional_encoder_15/add:z:0*
T0*,
_output_shapes
:���������(�
5layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_16/moments/meanMeanadd_10/add:z:0>layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(�
+layer_normalization_16/moments/StopGradientStopGradient,layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:���������(�
0layer_normalization_16/moments/SquaredDifferenceSquaredDifferenceadd_10/add:z:04layer_normalization_16/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������(��
9layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_16/moments/varianceMean4layer_normalization_16/moments/SquaredDifference:z:0Blayer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(k
&layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_16/batchnorm/addAddV20layer_normalization_16/moments/variance:output:0/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(�
&layer_normalization_16/batchnorm/RsqrtRsqrt(layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:���������(�
3layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_16/batchnorm/mulMul*layer_normalization_16/batchnorm/Rsqrt:y:0;layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_16/batchnorm/mul_1Muladd_10/add:z:0(layer_normalization_16/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_16/batchnorm/mul_2Mul,layer_normalization_16/moments/mean:output:0(layer_normalization_16/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_16/batchnorm/subSub7layer_normalization_16/batchnorm/ReadVariableOp:value:0*layer_normalization_16/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_16/batchnorm/add_1AddV2*layer_normalization_16/batchnorm/mul_1:z:0(layer_normalization_16/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(��
.sequential_5/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_10_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0n
$sequential_5/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_5/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_5/dense_10/Tensordot/ShapeShape*layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_5/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_5/dense_10/Tensordot/GatherV2GatherV2.sequential_5/dense_10/Tensordot/Shape:output:0-sequential_5/dense_10/Tensordot/free:output:06sequential_5/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_5/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_5/dense_10/Tensordot/GatherV2_1GatherV2.sequential_5/dense_10/Tensordot/Shape:output:0-sequential_5/dense_10/Tensordot/axes:output:08sequential_5/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_5/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
$sequential_5/dense_10/Tensordot/ProdProd1sequential_5/dense_10/Tensordot/GatherV2:output:0.sequential_5/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_5/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
&sequential_5/dense_10/Tensordot/Prod_1Prod3sequential_5/dense_10/Tensordot/GatherV2_1:output:00sequential_5/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_5/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&sequential_5/dense_10/Tensordot/concatConcatV2-sequential_5/dense_10/Tensordot/free:output:0-sequential_5/dense_10/Tensordot/axes:output:04sequential_5/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
%sequential_5/dense_10/Tensordot/stackPack-sequential_5/dense_10/Tensordot/Prod:output:0/sequential_5/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
)sequential_5/dense_10/Tensordot/transpose	Transpose*layer_normalization_16/batchnorm/add_1:z:0/sequential_5/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������(��
'sequential_5/dense_10/Tensordot/ReshapeReshape-sequential_5/dense_10/Tensordot/transpose:y:0.sequential_5/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
&sequential_5/dense_10/Tensordot/MatMulMatMul0sequential_5/dense_10/Tensordot/Reshape:output:06sequential_5/dense_10/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������r
'sequential_5/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�o
-sequential_5/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_5/dense_10/Tensordot/concat_1ConcatV21sequential_5/dense_10/Tensordot/GatherV2:output:00sequential_5/dense_10/Tensordot/Const_2:output:06sequential_5/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_5/dense_10/TensordotReshape0sequential_5/dense_10/Tensordot/MatMul:product:01sequential_5/dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������(��
,sequential_5/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_5/dense_10/BiasAddBiasAdd(sequential_5/dense_10/Tensordot:output:04sequential_5/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�e
 sequential_5/dense_10/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
sequential_5/dense_10/Gelu/mulMul)sequential_5/dense_10/Gelu/mul/x:output:0&sequential_5/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:���������(�f
!sequential_5/dense_10/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
"sequential_5/dense_10/Gelu/truedivRealDiv&sequential_5/dense_10/BiasAdd:output:0*sequential_5/dense_10/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������(��
sequential_5/dense_10/Gelu/ErfErf&sequential_5/dense_10/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������(�e
 sequential_5/dense_10/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential_5/dense_10/Gelu/addAddV2)sequential_5/dense_10/Gelu/add/x:output:0"sequential_5/dense_10/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������(��
 sequential_5/dense_10/Gelu/mul_1Mul"sequential_5/dense_10/Gelu/mul:z:0"sequential_5/dense_10/Gelu/add:z:0*
T0*,
_output_shapes
:���������(��

add_11/addAddV2$sequential_5/dense_10/Gelu/mul_1:z:0add_10/add:z:0*
T0*,
_output_shapes
:���������(�
5layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_17/moments/meanMeanadd_11/add:z:0>layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(�
+layer_normalization_17/moments/StopGradientStopGradient,layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:���������(�
0layer_normalization_17/moments/SquaredDifferenceSquaredDifferenceadd_11/add:z:04layer_normalization_17/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������(��
9layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_17/moments/varianceMean4layer_normalization_17/moments/SquaredDifference:z:0Blayer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(k
&layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_17/batchnorm/addAddV20layer_normalization_17/moments/variance:output:0/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(�
&layer_normalization_17/batchnorm/RsqrtRsqrt(layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:���������(�
3layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_17/batchnorm/mulMul*layer_normalization_17/batchnorm/Rsqrt:y:0;layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_17/batchnorm/mul_1Muladd_11/add:z:0(layer_normalization_17/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_17/batchnorm/mul_2Mul,layer_normalization_17/moments/mean:output:0(layer_normalization_17/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_17/batchnorm/subSub7layer_normalization_17/batchnorm/ReadVariableOp:value:0*layer_normalization_17/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_17/batchnorm/add_1AddV2*layer_normalization_17/batchnorm/mul_1:z:0(layer_normalization_17/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(�s
1global_average_pooling1d_5/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_average_pooling1d_5/MeanMean*layer_normalization_17/batchnorm/add_1:z:0:global_average_pooling1d_5/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:�����������
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_11/MatMulMatMul(global_average_pooling1d_5/Mean:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp0^layer_normalization_16/batchnorm/ReadVariableOp4^layer_normalization_16/batchnorm/mul/ReadVariableOp0^layer_normalization_17/batchnorm/ReadVariableOp4^layer_normalization_17/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp1^positional_encoder_15/embedding/embedding_lookup-^sequential_5/dense_10/BiasAdd/ReadVariableOp/^sequential_5/dense_10/Tensordot/ReadVariableOp6^tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp5^tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp6^tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp5^tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp6^tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp5^tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp6^tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp5^tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp6^tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp5^tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp6^tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp5^tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2b
/layer_normalization_15/batchnorm/ReadVariableOp/layer_normalization_15/batchnorm/ReadVariableOp2j
3layer_normalization_15/batchnorm/mul/ReadVariableOp3layer_normalization_15/batchnorm/mul/ReadVariableOp2b
/layer_normalization_16/batchnorm/ReadVariableOp/layer_normalization_16/batchnorm/ReadVariableOp2j
3layer_normalization_16/batchnorm/mul/ReadVariableOp3layer_normalization_16/batchnorm/mul/ReadVariableOp2b
/layer_normalization_17/batchnorm/ReadVariableOp/layer_normalization_17/batchnorm/ReadVariableOp2j
3layer_normalization_17/batchnorm/mul/ReadVariableOp3layer_normalization_17/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_5/attention_output/add/ReadVariableOp:multi_head_attention_5/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_5/key/add/ReadVariableOp-multi_head_attention_5/key/add/ReadVariableOp2r
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/query/add/ReadVariableOp/multi_head_attention_5/query/add/ReadVariableOp2v
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/value/add/ReadVariableOp/multi_head_attention_5/value/add/ReadVariableOp2v
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2d
0positional_encoder_15/embedding/embedding_lookup0positional_encoder_15/embedding/embedding_lookup2\
,sequential_5/dense_10/BiasAdd/ReadVariableOp,sequential_5/dense_10/BiasAdd/ReadVariableOp2`
.sequential_5/dense_10/Tensordot/ReadVariableOp.sequential_5/dense_10/Tensordot/ReadVariableOp2n
5tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp5tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp2l
4tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp4tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp2n
5tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp5tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp2l
4tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp4tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp2n
5tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp5tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp2l
4tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp4tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp2n
5tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp5tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp2l
4tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp4tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp2n
5tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp5tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp2l
4tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp4tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp2n
5tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp5tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp2l
4tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp4tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������

 
_user_specified_nameinputs: 

_output_shapes
:(
�(
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_385090

inputs>
*dense_10_tensordot_readvariableop_resource:
��7
(dense_10_biasadd_readvariableop_resource:	�
identity��dense_10/BiasAdd/ReadVariableOp�!dense_10/Tensordot/ReadVariableOp�
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0a
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_10/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_10/Tensordot/transpose	Transposeinputs"dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������(��
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�b
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������(��
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�X
dense_10/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dense_10/Gelu/mulMuldense_10/Gelu/mul/x:output:0dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:���������(�Y
dense_10/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
dense_10/Gelu/truedivRealDivdense_10/BiasAdd:output:0dense_10/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������(�j
dense_10/Gelu/ErfErfdense_10/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������(�X
dense_10/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dense_10/Gelu/addAddV2dense_10/Gelu/add/x:output:0dense_10/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������(�
dense_10/Gelu/mul_1Muldense_10/Gelu/mul:z:0dense_10/Gelu/add:z:0*
T0*,
_output_shapes
:���������(�k
IdentityIdentitydense_10/Gelu/mul_1:z:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�

�
Q__inference_positional_encoder_15_layer_call_and_return_conditional_losses_383116
encoded_tokens
unknown4
!embedding_embedding_lookup_383109:	(�
identity��embedding/embedding_lookup�
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_383109unknown*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/383109*
_output_shapes
:	(�*
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/383109*
_output_shapes
:	(��
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	(��
addAddV2encoded_tokens.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:���������(�[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:���������(�c
NoOpNoOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������(�:(: 28
embedding/embedding_lookupembedding/embedding_lookup:\ X
,
_output_shapes
:���������(�
(
_user_specified_nameencoded_tokens: 

_output_shapes
:(
��
�!
C__inference_model_5_layer_call_and_return_conditional_losses_384622

inputs[
=tubelet_embedding_30_conv3d_90_conv3d_readvariableop_resource: L
>tubelet_embedding_30_conv3d_90_biasadd_readvariableop_resource: [
=tubelet_embedding_30_conv3d_91_conv3d_readvariableop_resource: @L
>tubelet_embedding_30_conv3d_91_biasadd_readvariableop_resource:@\
=tubelet_embedding_30_conv3d_92_conv3d_readvariableop_resource:@�M
>tubelet_embedding_30_conv3d_92_biasadd_readvariableop_resource:	�[
=tubelet_embedding_31_conv3d_93_conv3d_readvariableop_resource: L
>tubelet_embedding_31_conv3d_93_biasadd_readvariableop_resource: [
=tubelet_embedding_31_conv3d_94_conv3d_readvariableop_resource: @L
>tubelet_embedding_31_conv3d_94_biasadd_readvariableop_resource:@\
=tubelet_embedding_31_conv3d_95_conv3d_readvariableop_resource:@�M
>tubelet_embedding_31_conv3d_95_biasadd_readvariableop_resource:	� 
positional_encoder_15_384478J
7positional_encoder_15_embedding_embedding_lookup_384480:	(�K
<layer_normalization_15_batchnorm_mul_readvariableop_resource:	�G
8layer_normalization_15_batchnorm_readvariableop_resource:	�Z
Bmulti_head_attention_5_query_einsum_einsum_readvariableop_resource:��K
8multi_head_attention_5_query_add_readvariableop_resource:	�X
@multi_head_attention_5_key_einsum_einsum_readvariableop_resource:��I
6multi_head_attention_5_key_add_readvariableop_resource:	�Z
Bmulti_head_attention_5_value_einsum_einsum_readvariableop_resource:��K
8multi_head_attention_5_value_add_readvariableop_resource:	�e
Mmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource:��R
Cmulti_head_attention_5_attention_output_add_readvariableop_resource:	�K
<layer_normalization_16_batchnorm_mul_readvariableop_resource:	�G
8layer_normalization_16_batchnorm_readvariableop_resource:	�K
7sequential_5_dense_10_tensordot_readvariableop_resource:
��D
5sequential_5_dense_10_biasadd_readvariableop_resource:	�K
<layer_normalization_17_batchnorm_mul_readvariableop_resource:	�G
8layer_normalization_17_batchnorm_readvariableop_resource:	�:
'dense_11_matmul_readvariableop_resource:	�6
(dense_11_biasadd_readvariableop_resource:
identity��dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�/layer_normalization_15/batchnorm/ReadVariableOp�3layer_normalization_15/batchnorm/mul/ReadVariableOp�/layer_normalization_16/batchnorm/ReadVariableOp�3layer_normalization_16/batchnorm/mul/ReadVariableOp�/layer_normalization_17/batchnorm/ReadVariableOp�3layer_normalization_17/batchnorm/mul/ReadVariableOp�:multi_head_attention_5/attention_output/add/ReadVariableOp�Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_5/key/add/ReadVariableOp�7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/query/add/ReadVariableOp�9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/value/add/ReadVariableOp�9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�0positional_encoder_15/embedding/embedding_lookup�,sequential_5/dense_10/BiasAdd/ReadVariableOp�.sequential_5/dense_10/Tensordot/ReadVariableOp�5tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp�4tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp�5tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp�4tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp�5tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp�4tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp�5tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp�4tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp�5tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp�4tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp�5tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp�4tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp�
/tf.__operators__.getitem_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_35/strided_sliceStridedSliceinputs8tf.__operators__.getitem_35/strided_slice/stack:output:0:tf.__operators__.getitem_35/strided_slice/stack_1:output:0:tf.__operators__.getitem_35/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
/tf.__operators__.getitem_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*)
value B"                    �
1tf.__operators__.getitem_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*)
value B"                   �
1tf.__operators__.getitem_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               �
)tf.__operators__.getitem_34/strided_sliceStridedSliceinputs8tf.__operators__.getitem_34/strided_slice/stack:output:0:tf.__operators__.getitem_34/strided_slice/stack_1:output:0:tf.__operators__.getitem_34/strided_slice/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:���������
*

begin_mask*
end_mask�
4tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_30_conv3d_90_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
%tubelet_embedding_30/conv3d_90/Conv3DConv3D2tf.__operators__.getitem_34/strided_slice:output:0<tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
5tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_30_conv3d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
&tubelet_embedding_30/conv3d_90/BiasAddBiasAdd.tubelet_embedding_30/conv3d_90/Conv3D:output:0=tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 �
#tubelet_embedding_30/conv3d_90/ReluRelu/tubelet_embedding_30/conv3d_90/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
/tubelet_embedding_30/max_pooling3d_60/MaxPool3D	MaxPool3D1tubelet_embedding_30/conv3d_90/Relu:activations:0*
T0*3
_output_shapes!
:���������
 *
ksize	
*
paddingVALID*
strides	
�
4tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_30_conv3d_91_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
%tubelet_embedding_30/conv3d_91/Conv3DConv3D8tubelet_embedding_30/max_pooling3d_60/MaxPool3D:output:0<tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@*
paddingSAME*
strides	
�
5tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_30_conv3d_91_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&tubelet_embedding_30/conv3d_91/BiasAddBiasAdd.tubelet_embedding_30/conv3d_91/Conv3D:output:0=tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@�
#tubelet_embedding_30/conv3d_91/ReluRelu/tubelet_embedding_30/conv3d_91/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
@�
/tubelet_embedding_30/max_pooling3d_61/MaxPool3D	MaxPool3D1tubelet_embedding_30/conv3d_91/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
4tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_30_conv3d_92_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
%tubelet_embedding_30/conv3d_92/Conv3DConv3D8tubelet_embedding_30/max_pooling3d_61/MaxPool3D:output:0<tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
5tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_30_conv3d_92_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&tubelet_embedding_30/conv3d_92/BiasAddBiasAdd.tubelet_embedding_30/conv3d_92/Conv3D:output:0=tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
3tubelet_embedding_30/average_pooling3d_30/AvgPool3D	AvgPool3D/tubelet_embedding_30/conv3d_92/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
�*
ksize	
*
paddingVALID*
strides	
�
%tubelet_embedding_30/reshape_30/ShapeShape<tubelet_embedding_30/average_pooling3d_30/AvgPool3D:output:0*
T0*
_output_shapes
:}
3tubelet_embedding_30/reshape_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tubelet_embedding_30/reshape_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tubelet_embedding_30/reshape_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-tubelet_embedding_30/reshape_30/strided_sliceStridedSlice.tubelet_embedding_30/reshape_30/Shape:output:0<tubelet_embedding_30/reshape_30/strided_slice/stack:output:0>tubelet_embedding_30/reshape_30/strided_slice/stack_1:output:0>tubelet_embedding_30/reshape_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/tubelet_embedding_30/reshape_30/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������r
/tubelet_embedding_30/reshape_30/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
-tubelet_embedding_30/reshape_30/Reshape/shapePack6tubelet_embedding_30/reshape_30/strided_slice:output:08tubelet_embedding_30/reshape_30/Reshape/shape/1:output:08tubelet_embedding_30/reshape_30/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
'tubelet_embedding_30/reshape_30/ReshapeReshape<tubelet_embedding_30/average_pooling3d_30/AvgPool3D:output:06tubelet_embedding_30/reshape_30/Reshape/shape:output:0*
T0*,
_output_shapes
:���������(��
4tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_31_conv3d_93_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
%tubelet_embedding_31/conv3d_93/Conv3DConv3D2tf.__operators__.getitem_35/strided_slice:output:0<tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
5tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_31_conv3d_93_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
&tubelet_embedding_31/conv3d_93/BiasAddBiasAdd.tubelet_embedding_31/conv3d_93/Conv3D:output:0=tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 �
#tubelet_embedding_31/conv3d_93/ReluRelu/tubelet_embedding_31/conv3d_93/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
/tubelet_embedding_31/max_pooling3d_62/MaxPool3D	MaxPool3D1tubelet_embedding_31/conv3d_93/Relu:activations:0*
T0*3
_output_shapes!
:���������
 *
ksize	
*
paddingVALID*
strides	
�
4tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_31_conv3d_94_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
%tubelet_embedding_31/conv3d_94/Conv3DConv3D8tubelet_embedding_31/max_pooling3d_62/MaxPool3D:output:0<tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@*
paddingSAME*
strides	
�
5tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_31_conv3d_94_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&tubelet_embedding_31/conv3d_94/BiasAddBiasAdd.tubelet_embedding_31/conv3d_94/Conv3D:output:0=tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@�
#tubelet_embedding_31/conv3d_94/ReluRelu/tubelet_embedding_31/conv3d_94/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
@�
/tubelet_embedding_31/max_pooling3d_63/MaxPool3D	MaxPool3D1tubelet_embedding_31/conv3d_94/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
4tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOpReadVariableOp=tubelet_embedding_31_conv3d_95_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
%tubelet_embedding_31/conv3d_95/Conv3DConv3D8tubelet_embedding_31/max_pooling3d_63/MaxPool3D:output:0<tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
5tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOpReadVariableOp>tubelet_embedding_31_conv3d_95_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&tubelet_embedding_31/conv3d_95/BiasAddBiasAdd.tubelet_embedding_31/conv3d_95/Conv3D:output:0=tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
3tubelet_embedding_31/average_pooling3d_31/AvgPool3D	AvgPool3D/tubelet_embedding_31/conv3d_95/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
�*
ksize	
*
paddingVALID*
strides	
�
%tubelet_embedding_31/reshape_31/ShapeShape<tubelet_embedding_31/average_pooling3d_31/AvgPool3D:output:0*
T0*
_output_shapes
:}
3tubelet_embedding_31/reshape_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5tubelet_embedding_31/reshape_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5tubelet_embedding_31/reshape_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-tubelet_embedding_31/reshape_31/strided_sliceStridedSlice.tubelet_embedding_31/reshape_31/Shape:output:0<tubelet_embedding_31/reshape_31/strided_slice/stack:output:0>tubelet_embedding_31/reshape_31/strided_slice/stack_1:output:0>tubelet_embedding_31/reshape_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/tubelet_embedding_31/reshape_31/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������r
/tubelet_embedding_31/reshape_31/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
-tubelet_embedding_31/reshape_31/Reshape/shapePack6tubelet_embedding_31/reshape_31/strided_slice:output:08tubelet_embedding_31/reshape_31/Reshape/shape/1:output:08tubelet_embedding_31/reshape_31/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
'tubelet_embedding_31/reshape_31/ReshapeReshape<tubelet_embedding_31/average_pooling3d_31/AvgPool3D:output:06tubelet_embedding_31/reshape_31/Reshape/shape:output:0*
T0*,
_output_shapes
:���������(�\
concatenate_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_12/concatConcatV20tubelet_embedding_30/reshape_30/Reshape:output:00tubelet_embedding_31/reshape_31/Reshape:output:0#concatenate_12/concat/axis:output:0*
N*
T0*,
_output_shapes
:���������(��
0positional_encoder_15/embedding/embedding_lookupResourceGather7positional_encoder_15_embedding_embedding_lookup_384480positional_encoder_15_384478*
Tindices0*J
_class@
><loc:@positional_encoder_15/embedding/embedding_lookup/384480*
_output_shapes
:	(�*
dtype0�
9positional_encoder_15/embedding/embedding_lookup/IdentityIdentity9positional_encoder_15/embedding/embedding_lookup:output:0*
T0*J
_class@
><loc:@positional_encoder_15/embedding/embedding_lookup/384480*
_output_shapes
:	(��
;positional_encoder_15/embedding/embedding_lookup/Identity_1IdentityBpositional_encoder_15/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	(��
positional_encoder_15/addAddV2concatenate_12/concat:output:0Dpositional_encoder_15/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:���������(�
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_15/moments/meanMeanpositional_encoder_15/add:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(�
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:���������(�
0layer_normalization_15/moments/SquaredDifferenceSquaredDifferencepositional_encoder_15/add:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������(��
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(k
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(�
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:���������(�
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_15/batchnorm/mul_1Mulpositional_encoder_15/add:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(��
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
*multi_head_attention_5/query/einsum/EinsumEinsum*layer_normalization_15/batchnorm/add_1:z:0Amulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde�
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
(multi_head_attention_5/key/einsum/EinsumEinsum*layer_normalization_15/batchnorm/add_1:z:0?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde�
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
*multi_head_attention_5/value/einsum/EinsumEinsum*layer_normalization_15/batchnorm/add_1:z:0Amulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde�
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(�a
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=�
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*0
_output_shapes
:���������(��
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:���������((*
equationaecd,abcd->acbe�
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������((q
,multi_head_attention_5/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
*multi_head_attention_5/dropout/dropout/MulMul0multi_head_attention_5/softmax/Softmax:softmax:05multi_head_attention_5/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������((�
,multi_head_attention_5/dropout/dropout/ShapeShape0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*
_output_shapes
:�
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_5/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������((*
dtype0*

seed*z
5multi_head_attention_5/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
3multi_head_attention_5/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_5/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������((�
+multi_head_attention_5/dropout/dropout/CastCast7multi_head_attention_5/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������((�
,multi_head_attention_5/dropout/dropout/Mul_1Mul.multi_head_attention_5/dropout/dropout/Mul:z:0/multi_head_attention_5/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������((�
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/dropout/Mul_1:z:0$multi_head_attention_5/value/add:z:0*
N*
T0*0
_output_shapes
:���������(�*
equationacbe,aecd->abcd�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������(�*
equationabcd,cde->abe�
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��

add_10/addAddV2/multi_head_attention_5/attention_output/add:z:0positional_encoder_15/add:z:0*
T0*,
_output_shapes
:���������(�
5layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_16/moments/meanMeanadd_10/add:z:0>layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(�
+layer_normalization_16/moments/StopGradientStopGradient,layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:���������(�
0layer_normalization_16/moments/SquaredDifferenceSquaredDifferenceadd_10/add:z:04layer_normalization_16/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������(��
9layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_16/moments/varianceMean4layer_normalization_16/moments/SquaredDifference:z:0Blayer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(k
&layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_16/batchnorm/addAddV20layer_normalization_16/moments/variance:output:0/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(�
&layer_normalization_16/batchnorm/RsqrtRsqrt(layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:���������(�
3layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_16/batchnorm/mulMul*layer_normalization_16/batchnorm/Rsqrt:y:0;layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_16/batchnorm/mul_1Muladd_10/add:z:0(layer_normalization_16/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_16/batchnorm/mul_2Mul,layer_normalization_16/moments/mean:output:0(layer_normalization_16/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_16/batchnorm/subSub7layer_normalization_16/batchnorm/ReadVariableOp:value:0*layer_normalization_16/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_16/batchnorm/add_1AddV2*layer_normalization_16/batchnorm/mul_1:z:0(layer_normalization_16/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(��
.sequential_5/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_10_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0n
$sequential_5/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_5/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
%sequential_5/dense_10/Tensordot/ShapeShape*layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:o
-sequential_5/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_5/dense_10/Tensordot/GatherV2GatherV2.sequential_5/dense_10/Tensordot/Shape:output:0-sequential_5/dense_10/Tensordot/free:output:06sequential_5/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_5/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_5/dense_10/Tensordot/GatherV2_1GatherV2.sequential_5/dense_10/Tensordot/Shape:output:0-sequential_5/dense_10/Tensordot/axes:output:08sequential_5/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_5/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
$sequential_5/dense_10/Tensordot/ProdProd1sequential_5/dense_10/Tensordot/GatherV2:output:0.sequential_5/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_5/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
&sequential_5/dense_10/Tensordot/Prod_1Prod3sequential_5/dense_10/Tensordot/GatherV2_1:output:00sequential_5/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_5/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&sequential_5/dense_10/Tensordot/concatConcatV2-sequential_5/dense_10/Tensordot/free:output:0-sequential_5/dense_10/Tensordot/axes:output:04sequential_5/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
%sequential_5/dense_10/Tensordot/stackPack-sequential_5/dense_10/Tensordot/Prod:output:0/sequential_5/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
)sequential_5/dense_10/Tensordot/transpose	Transpose*layer_normalization_16/batchnorm/add_1:z:0/sequential_5/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������(��
'sequential_5/dense_10/Tensordot/ReshapeReshape-sequential_5/dense_10/Tensordot/transpose:y:0.sequential_5/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
&sequential_5/dense_10/Tensordot/MatMulMatMul0sequential_5/dense_10/Tensordot/Reshape:output:06sequential_5/dense_10/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������r
'sequential_5/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�o
-sequential_5/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_5/dense_10/Tensordot/concat_1ConcatV21sequential_5/dense_10/Tensordot/GatherV2:output:00sequential_5/dense_10/Tensordot/Const_2:output:06sequential_5/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_5/dense_10/TensordotReshape0sequential_5/dense_10/Tensordot/MatMul:product:01sequential_5/dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������(��
,sequential_5/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_5/dense_10/BiasAddBiasAdd(sequential_5/dense_10/Tensordot:output:04sequential_5/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�e
 sequential_5/dense_10/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
sequential_5/dense_10/Gelu/mulMul)sequential_5/dense_10/Gelu/mul/x:output:0&sequential_5/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:���������(�f
!sequential_5/dense_10/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
"sequential_5/dense_10/Gelu/truedivRealDiv&sequential_5/dense_10/BiasAdd:output:0*sequential_5/dense_10/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������(��
sequential_5/dense_10/Gelu/ErfErf&sequential_5/dense_10/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������(�e
 sequential_5/dense_10/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sequential_5/dense_10/Gelu/addAddV2)sequential_5/dense_10/Gelu/add/x:output:0"sequential_5/dense_10/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������(��
 sequential_5/dense_10/Gelu/mul_1Mul"sequential_5/dense_10/Gelu/mul:z:0"sequential_5/dense_10/Gelu/add:z:0*
T0*,
_output_shapes
:���������(��

add_11/addAddV2$sequential_5/dense_10/Gelu/mul_1:z:0add_10/add:z:0*
T0*,
_output_shapes
:���������(�
5layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_17/moments/meanMeanadd_11/add:z:0>layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(�
+layer_normalization_17/moments/StopGradientStopGradient,layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:���������(�
0layer_normalization_17/moments/SquaredDifferenceSquaredDifferenceadd_11/add:z:04layer_normalization_17/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������(��
9layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'layer_normalization_17/moments/varianceMean4layer_normalization_17/moments/SquaredDifference:z:0Blayer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(k
&layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
$layer_normalization_17/batchnorm/addAddV20layer_normalization_17/moments/variance:output:0/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(�
&layer_normalization_17/batchnorm/RsqrtRsqrt(layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:���������(�
3layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_17/batchnorm/mulMul*layer_normalization_17/batchnorm/Rsqrt:y:0;layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_17/batchnorm/mul_1Muladd_11/add:z:0(layer_normalization_17/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_17/batchnorm/mul_2Mul,layer_normalization_17/moments/mean:output:0(layer_normalization_17/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(��
/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$layer_normalization_17/batchnorm/subSub7layer_normalization_17/batchnorm/ReadVariableOp:value:0*layer_normalization_17/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(��
&layer_normalization_17/batchnorm/add_1AddV2*layer_normalization_17/batchnorm/mul_1:z:0(layer_normalization_17/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(�s
1global_average_pooling1d_5/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_average_pooling1d_5/MeanMean*layer_normalization_17/batchnorm/add_1:z:0:global_average_pooling1d_5/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:�����������
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_11/MatMulMatMul(global_average_pooling1d_5/Mean:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp0^layer_normalization_16/batchnorm/ReadVariableOp4^layer_normalization_16/batchnorm/mul/ReadVariableOp0^layer_normalization_17/batchnorm/ReadVariableOp4^layer_normalization_17/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp1^positional_encoder_15/embedding/embedding_lookup-^sequential_5/dense_10/BiasAdd/ReadVariableOp/^sequential_5/dense_10/Tensordot/ReadVariableOp6^tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp5^tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp6^tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp5^tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp6^tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp5^tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp6^tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp5^tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp6^tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp5^tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp6^tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp5^tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������
: : : : : : : : : : : : :(: : : : : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2b
/layer_normalization_15/batchnorm/ReadVariableOp/layer_normalization_15/batchnorm/ReadVariableOp2j
3layer_normalization_15/batchnorm/mul/ReadVariableOp3layer_normalization_15/batchnorm/mul/ReadVariableOp2b
/layer_normalization_16/batchnorm/ReadVariableOp/layer_normalization_16/batchnorm/ReadVariableOp2j
3layer_normalization_16/batchnorm/mul/ReadVariableOp3layer_normalization_16/batchnorm/mul/ReadVariableOp2b
/layer_normalization_17/batchnorm/ReadVariableOp/layer_normalization_17/batchnorm/ReadVariableOp2j
3layer_normalization_17/batchnorm/mul/ReadVariableOp3layer_normalization_17/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_5/attention_output/add/ReadVariableOp:multi_head_attention_5/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_5/key/add/ReadVariableOp-multi_head_attention_5/key/add/ReadVariableOp2r
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/query/add/ReadVariableOp/multi_head_attention_5/query/add/ReadVariableOp2v
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/value/add/ReadVariableOp/multi_head_attention_5/value/add/ReadVariableOp2v
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2d
0positional_encoder_15/embedding/embedding_lookup0positional_encoder_15/embedding/embedding_lookup2\
,sequential_5/dense_10/BiasAdd/ReadVariableOp,sequential_5/dense_10/BiasAdd/ReadVariableOp2`
.sequential_5/dense_10/Tensordot/ReadVariableOp.sequential_5/dense_10/Tensordot/ReadVariableOp2n
5tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp5tubelet_embedding_30/conv3d_90/BiasAdd/ReadVariableOp2l
4tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp4tubelet_embedding_30/conv3d_90/Conv3D/ReadVariableOp2n
5tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp5tubelet_embedding_30/conv3d_91/BiasAdd/ReadVariableOp2l
4tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp4tubelet_embedding_30/conv3d_91/Conv3D/ReadVariableOp2n
5tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp5tubelet_embedding_30/conv3d_92/BiasAdd/ReadVariableOp2l
4tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp4tubelet_embedding_30/conv3d_92/Conv3D/ReadVariableOp2n
5tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp5tubelet_embedding_31/conv3d_93/BiasAdd/ReadVariableOp2l
4tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp4tubelet_embedding_31/conv3d_93/Conv3D/ReadVariableOp2n
5tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp5tubelet_embedding_31/conv3d_94/BiasAdd/ReadVariableOp2l
4tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp4tubelet_embedding_31/conv3d_94/Conv3D/ReadVariableOp2n
5tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp5tubelet_embedding_31/conv3d_95/BiasAdd/ReadVariableOp2l
4tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp4tubelet_embedding_31/conv3d_95/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������

 
_user_specified_nameinputs: 

_output_shapes
:(
�
�
7__inference_layer_normalization_17_layer_call_fn_385149

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_layer_normalization_17_layer_call_and_return_conditional_losses_383276t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
�
-__inference_sequential_5_layer_call_fn_385043

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������(�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_382896t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�"
�
D__inference_dense_10_layer_call_and_return_conditional_losses_385308

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
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
value	B : �
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
value	B : �
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
value	B : �
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
:���������(��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������(�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*,
_output_shapes
:���������(�P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?v
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������(�X
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:���������(�O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:���������(�d

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:���������(�b
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*,
_output_shapes
:���������(�z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�
l
P__inference_average_pooling3d_30_layer_call_and_return_conditional_losses_385231

inputs
identity�
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling3d_61_layer_call_and_return_conditional_losses_385221

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_5_layer_call_fn_384888	
query	
value
unknown:��
	unknown_0:	�!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:���������(�:���������((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_383186t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������((`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:���������(�:���������(�: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:���������(�

_user_specified_namequery:SO
,
_output_shapes
:���������(�

_user_specified_namevalue
�+
�
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_384948	
query	
valueC
+query_einsum_einsum_readvariableop_resource:��4
!query_add_readvariableop_resource:	�A
)key_einsum_einsum_readvariableop_resource:��2
key_add_readvariableop_resource:	�C
+value_einsum_einsum_readvariableop_resource:��4
!value_add_readvariableop_resource:	�N
6attention_output_einsum_einsum_readvariableop_resource:��;
,attention_output_add_readvariableop_resource:	�
identity

identity_1��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde{
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abdew
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(��
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:���������(�*
equationabc,cde->abde{
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes
:	�*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������(�J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��=d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:���������(��
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������((*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������((q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������((�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:���������(�*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:��*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������(�*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:���������(�r

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������((�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:���������(�:���������(�: : : : : : : : 2J
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
:���������(�

_user_specified_namequery:SO
,
_output_shapes
:���������(�

_user_specified_namevalue
�
Q
5__inference_average_pooling3d_30_layer_call_fn_385226

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_average_pooling3d_30_layer_call_and_return_conditional_losses_382805�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
R__inference_layer_normalization_16_layer_call_and_return_conditional_losses_383235

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������(�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������(�l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������(*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������(a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������(
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������(�h
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������(�w
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������(�g
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������(�: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������(�
 
_user_specified_nameinputs
�-
�
P__inference_tubelet_embedding_31_layer_call_and_return_conditional_losses_383081

videosF
(conv3d_93_conv3d_readvariableop_resource: 7
)conv3d_93_biasadd_readvariableop_resource: F
(conv3d_94_conv3d_readvariableop_resource: @7
)conv3d_94_biasadd_readvariableop_resource:@G
(conv3d_95_conv3d_readvariableop_resource:@�8
)conv3d_95_biasadd_readvariableop_resource:	�
identity�� conv3d_93/BiasAdd/ReadVariableOp�conv3d_93/Conv3D/ReadVariableOp� conv3d_94/BiasAdd/ReadVariableOp�conv3d_94/Conv3D/ReadVariableOp� conv3d_95/BiasAdd/ReadVariableOp�conv3d_95/Conv3D/ReadVariableOp�
conv3d_93/Conv3D/ReadVariableOpReadVariableOp(conv3d_93_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
conv3d_93/Conv3DConv3Dvideos'conv3d_93/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
 conv3d_93/BiasAdd/ReadVariableOpReadVariableOp)conv3d_93_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv3d_93/BiasAddBiasAddconv3d_93/Conv3D:output:0(conv3d_93/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 p
conv3d_93/ReluReluconv3d_93/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
max_pooling3d_62/MaxPool3D	MaxPool3Dconv3d_93/Relu:activations:0*
T0*3
_output_shapes!
:���������
 *
ksize	
*
paddingVALID*
strides	
�
conv3d_94/Conv3D/ReadVariableOpReadVariableOp(conv3d_94_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
conv3d_94/Conv3DConv3D#max_pooling3d_62/MaxPool3D:output:0'conv3d_94/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@*
paddingSAME*
strides	
�
 conv3d_94/BiasAdd/ReadVariableOpReadVariableOp)conv3d_94_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d_94/BiasAddBiasAddconv3d_94/Conv3D:output:0(conv3d_94/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
@p
conv3d_94/ReluReluconv3d_94/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
@�
max_pooling3d_63/MaxPool3D	MaxPool3Dconv3d_94/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
conv3d_95/Conv3D/ReadVariableOpReadVariableOp(conv3d_95_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
conv3d_95/Conv3DConv3D#max_pooling3d_63/MaxPool3D:output:0'conv3d_95/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
 conv3d_95/BiasAdd/ReadVariableOpReadVariableOp)conv3d_95_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_95/BiasAddBiasAddconv3d_95/Conv3D:output:0(conv3d_95/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
average_pooling3d_31/AvgPool3D	AvgPool3Dconv3d_95/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
�*
ksize	
*
paddingVALID*
strides	
g
reshape_31/ShapeShape'average_pooling3d_31/AvgPool3D:output:0*
T0*
_output_shapes
:h
reshape_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_31/strided_sliceStridedSlicereshape_31/Shape:output:0'reshape_31/strided_slice/stack:output:0)reshape_31/strided_slice/stack_1:output:0)reshape_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
reshape_31/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������]
reshape_31/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_31/Reshape/shapePack!reshape_31/strided_slice:output:0#reshape_31/Reshape/shape/1:output:0#reshape_31/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
reshape_31/ReshapeReshape'average_pooling3d_31/AvgPool3D:output:0!reshape_31/Reshape/shape:output:0*
T0*,
_output_shapes
:���������(�o
IdentityIdentityreshape_31/Reshape:output:0^NoOp*
T0*,
_output_shapes
:���������(��
NoOpNoOp!^conv3d_93/BiasAdd/ReadVariableOp ^conv3d_93/Conv3D/ReadVariableOp!^conv3d_94/BiasAdd/ReadVariableOp ^conv3d_94/Conv3D/ReadVariableOp!^conv3d_95/BiasAdd/ReadVariableOp ^conv3d_95/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������
: : : : : : 2D
 conv3d_93/BiasAdd/ReadVariableOp conv3d_93/BiasAdd/ReadVariableOp2B
conv3d_93/Conv3D/ReadVariableOpconv3d_93/Conv3D/ReadVariableOp2D
 conv3d_94/BiasAdd/ReadVariableOp conv3d_94/BiasAdd/ReadVariableOp2B
conv3d_94/Conv3D/ReadVariableOpconv3d_94/Conv3D/ReadVariableOp2D
 conv3d_95/BiasAdd/ReadVariableOp conv3d_95/BiasAdd/ReadVariableOp2B
conv3d_95/Conv3D/ReadVariableOpconv3d_95/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������

 
_user_specified_namevideos
�
Q
5__inference_average_pooling3d_31_layer_call_fn_385256

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *Y
fTRR
P__inference_average_pooling3d_31_layer_call_and_return_conditional_losses_382841�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
W
;__inference_global_average_pooling1d_5_layer_call_fn_385176

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *_
fZRX
V__inference_global_average_pooling1d_5_layer_call_and_return_conditional_losses_382977i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_5_layer_call_fn_384912	
query	
value
unknown:��
	unknown_0:	�!
	unknown_1:��
	unknown_2:	�!
	unknown_3:��
	unknown_4:	�!
	unknown_5:��
	unknown_6:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *G
_output_shapes5
3:���������(�:���������((**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2*0,1J 8� *[
fVRT
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_383482t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������(�y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������((`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:���������(�:���������(�: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:���������(�

_user_specified_namequery:SO
,
_output_shapes
:���������(�

_user_specified_namevalue
�
r
V__inference_global_average_pooling1d_5_layer_call_and_return_conditional_losses_385182

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
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
input_16=
serving_default_input_16:0���������
<
dense_110
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
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
�
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
�
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
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
�
<position_embedding
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
�
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
�
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
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
�
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
�
glayer_with_weights-0
glayer-0
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
�
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
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�iter
�beta_1
�beta_2

�decay
�learning_rateDm�Em�_m�`m�um�vm�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�Dv�Ev�_v�`v�uv�vv�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
D13
E14
�15
�16
�17
�18
�19
�20
�21
�22
_23
`24
�25
�26
u27
v28
�29
�30"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
D13
E14
�15
�16
�17
�18
�19
�20
�21
�22
_23
`24
�25
�26
u27
v28
�29
�30"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_model_5_layer_call_fn_383367
(__inference_model_5_layer_call_fn_384118
(__inference_model_5_layer_call_fn_384187
(__inference_model_5_layer_call_fn_383863�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_model_5_layer_call_and_return_conditional_losses_384401
C__inference_model_5_layer_call_and_return_conditional_losses_384622
C__inference_model_5_layer_call_and_return_conditional_losses_383953
C__inference_model_5_layer_call_and_return_conditional_losses_384043�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
!__inference__wrapped_model_382772input_16"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_tubelet_embedding_30_layer_call_fn_384710�
���
FullArgSpec
args�
jself
jvideos
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_tubelet_embedding_30_layer_call_and_return_conditional_losses_384746�
���
FullArgSpec
args�
jself
jvideos
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_tubelet_embedding_31_layer_call_fn_384763�
���
FullArgSpec
args�
jself
jvideos
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_tubelet_embedding_31_layer_call_and_return_conditional_losses_384799�
���
FullArgSpec
args�
jself
jvideos
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_concatenate_12_layer_call_fn_384805�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_concatenate_12_layer_call_and_return_conditional_losses_384812�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
embeddings
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�2�
6__inference_positional_encoder_15_layer_call_fn_384821�
���
FullArgSpec%
args�
jself
jencoded_tokens
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_positional_encoder_15_layer_call_and_return_conditional_losses_384833�
���
FullArgSpec%
args�
jself
jencoded_tokens
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
+:)�2layer_normalization_15/gamma
*:(�2layer_normalization_15/beta
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�2�
7__inference_layer_normalization_15_layer_call_fn_384842�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_384864�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
�2�
7__inference_multi_head_attention_5_layer_call_fn_384888
7__inference_multi_head_attention_5_layer_call_fn_384912�
���
FullArgSpece
args]�Z
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
defaults�

 

 
p 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_384948
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_384991�
���
FullArgSpece
args]�Z
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
defaults�

 

 
p 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_add_10_layer_call_fn_384997�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_add_10_layer_call_and_return_conditional_losses_385003�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
+:)�2layer_normalization_16/gamma
*:(�2layer_normalization_16/beta
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�2�
7__inference_layer_normalization_16_layer_call_fn_385012�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_layer_normalization_16_layer_call_and_return_conditional_losses_385034�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_sequential_5_layer_call_fn_382903
-__inference_sequential_5_layer_call_fn_385043
-__inference_sequential_5_layer_call_fn_385052
-__inference_sequential_5_layer_call_fn_382949�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_sequential_5_layer_call_and_return_conditional_losses_385090
H__inference_sequential_5_layer_call_and_return_conditional_losses_385128
H__inference_sequential_5_layer_call_and_return_conditional_losses_382958
H__inference_sequential_5_layer_call_and_return_conditional_losses_382967�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_add_11_layer_call_fn_385134�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_add_11_layer_call_and_return_conditional_losses_385140�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
+:)�2layer_normalization_17/gamma
*:(�2layer_normalization_17/beta
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�2�
7__inference_layer_normalization_17_layer_call_fn_385149�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_layer_normalization_17_layer_call_and_return_conditional_losses_385171�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
;__inference_global_average_pooling1d_5_layer_call_fn_385176�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
V__inference_global_average_pooling1d_5_layer_call_and_return_conditional_losses_385182�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
": 	�2dense_11/kernel
:2dense_11/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_dense_11_layer_call_fn_385191�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_11_layer_call_and_return_conditional_losses_385201�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
C:A 2%tubelet_embedding_30/conv3d_90/kernel
1:/ 2#tubelet_embedding_30/conv3d_90/bias
C:A @2%tubelet_embedding_30/conv3d_91/kernel
1:/@2#tubelet_embedding_30/conv3d_91/bias
D:B@�2%tubelet_embedding_30/conv3d_92/kernel
2:0�2#tubelet_embedding_30/conv3d_92/bias
C:A 2%tubelet_embedding_31/conv3d_93/kernel
1:/ 2#tubelet_embedding_31/conv3d_93/bias
C:A @2%tubelet_embedding_31/conv3d_94/kernel
1:/@2#tubelet_embedding_31/conv3d_94/bias
D:B@�2%tubelet_embedding_31/conv3d_95/kernel
2:0�2#tubelet_embedding_31/conv3d_95/bias
=:;	(�2*positional_encoder_15/embedding/embeddings
;:9��2#multi_head_attention_5/query/kernel
4:2	�2!multi_head_attention_5/query/bias
9:7��2!multi_head_attention_5/key/kernel
2:0	�2multi_head_attention_5/key/bias
;:9��2#multi_head_attention_5/value/kernel
4:2	�2!multi_head_attention_5/value/bias
F:D��2.multi_head_attention_5/attention_output/kernel
;:9�2,multi_head_attention_5/attention_output/bias
#:!
��2dense_10/kernel
:�2dense_10/bias
 "
trackable_list_wrapper
�
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
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_signature_wrapper_384693input_16"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
1__inference_max_pooling3d_60_layer_call_fn_385206�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_max_pooling3d_60_layer_call_and_return_conditional_losses_385211�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
1__inference_max_pooling3d_61_layer_call_fn_385216�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_max_pooling3d_61_layer_call_and_return_conditional_losses_385221�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_average_pooling3d_30_layer_call_fn_385226�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_average_pooling3d_30_layer_call_and_return_conditional_losses_385231�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
1__inference_max_pooling3d_62_layer_call_fn_385236�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_max_pooling3d_62_layer_call_and_return_conditional_losses_385241�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
1__inference_max_pooling3d_63_layer_call_fn_385246�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_max_pooling3d_63_layer_call_and_return_conditional_losses_385251�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_average_pooling3d_31_layer_call_fn_385256�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_average_pooling3d_31_layer_call_and_return_conditional_losses_385261�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_dense_10_layer_call_fn_385270�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_10_layer_call_and_return_conditional_losses_385308�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
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
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
0:.�2#Adam/layer_normalization_15/gamma/m
/:-�2"Adam/layer_normalization_15/beta/m
0:.�2#Adam/layer_normalization_16/gamma/m
/:-�2"Adam/layer_normalization_16/beta/m
0:.�2#Adam/layer_normalization_17/gamma/m
/:-�2"Adam/layer_normalization_17/beta/m
':%	�2Adam/dense_11/kernel/m
 :2Adam/dense_11/bias/m
H:F 2,Adam/tubelet_embedding_30/conv3d_90/kernel/m
6:4 2*Adam/tubelet_embedding_30/conv3d_90/bias/m
H:F @2,Adam/tubelet_embedding_30/conv3d_91/kernel/m
6:4@2*Adam/tubelet_embedding_30/conv3d_91/bias/m
I:G@�2,Adam/tubelet_embedding_30/conv3d_92/kernel/m
7:5�2*Adam/tubelet_embedding_30/conv3d_92/bias/m
H:F 2,Adam/tubelet_embedding_31/conv3d_93/kernel/m
6:4 2*Adam/tubelet_embedding_31/conv3d_93/bias/m
H:F @2,Adam/tubelet_embedding_31/conv3d_94/kernel/m
6:4@2*Adam/tubelet_embedding_31/conv3d_94/bias/m
I:G@�2,Adam/tubelet_embedding_31/conv3d_95/kernel/m
7:5�2*Adam/tubelet_embedding_31/conv3d_95/bias/m
B:@	(�21Adam/positional_encoder_15/embedding/embeddings/m
@:>��2*Adam/multi_head_attention_5/query/kernel/m
9:7	�2(Adam/multi_head_attention_5/query/bias/m
>:<��2(Adam/multi_head_attention_5/key/kernel/m
7:5	�2&Adam/multi_head_attention_5/key/bias/m
@:>��2*Adam/multi_head_attention_5/value/kernel/m
9:7	�2(Adam/multi_head_attention_5/value/bias/m
K:I��25Adam/multi_head_attention_5/attention_output/kernel/m
@:>�23Adam/multi_head_attention_5/attention_output/bias/m
(:&
��2Adam/dense_10/kernel/m
!:�2Adam/dense_10/bias/m
0:.�2#Adam/layer_normalization_15/gamma/v
/:-�2"Adam/layer_normalization_15/beta/v
0:.�2#Adam/layer_normalization_16/gamma/v
/:-�2"Adam/layer_normalization_16/beta/v
0:.�2#Adam/layer_normalization_17/gamma/v
/:-�2"Adam/layer_normalization_17/beta/v
':%	�2Adam/dense_11/kernel/v
 :2Adam/dense_11/bias/v
H:F 2,Adam/tubelet_embedding_30/conv3d_90/kernel/v
6:4 2*Adam/tubelet_embedding_30/conv3d_90/bias/v
H:F @2,Adam/tubelet_embedding_30/conv3d_91/kernel/v
6:4@2*Adam/tubelet_embedding_30/conv3d_91/bias/v
I:G@�2,Adam/tubelet_embedding_30/conv3d_92/kernel/v
7:5�2*Adam/tubelet_embedding_30/conv3d_92/bias/v
H:F 2,Adam/tubelet_embedding_31/conv3d_93/kernel/v
6:4 2*Adam/tubelet_embedding_31/conv3d_93/bias/v
H:F @2,Adam/tubelet_embedding_31/conv3d_94/kernel/v
6:4@2*Adam/tubelet_embedding_31/conv3d_94/bias/v
I:G@�2,Adam/tubelet_embedding_31/conv3d_95/kernel/v
7:5�2*Adam/tubelet_embedding_31/conv3d_95/bias/v
B:@	(�21Adam/positional_encoder_15/embedding/embeddings/v
@:>��2*Adam/multi_head_attention_5/query/kernel/v
9:7	�2(Adam/multi_head_attention_5/query/bias/v
>:<��2(Adam/multi_head_attention_5/key/kernel/v
7:5	�2&Adam/multi_head_attention_5/key/bias/v
@:>��2*Adam/multi_head_attention_5/value/kernel/v
9:7	�2(Adam/multi_head_attention_5/value/bias/v
K:I��25Adam/multi_head_attention_5/attention_output/kernel/v
@:>�23Adam/multi_head_attention_5/attention_output/bias/v
(:&
��2Adam/dense_10/kernel/v
!:�2Adam/dense_10/bias/v
	J
Const�
!__inference__wrapped_model_382772�:��������������DE��������_`��uv��=�:
3�0
.�+
input_16���������

� "3�0
.
dense_11"�
dense_11����������
B__inference_add_10_layer_call_and_return_conditional_losses_385003�d�a
Z�W
U�R
'�$
inputs/0���������(�
'�$
inputs/1���������(�
� "*�'
 �
0���������(�
� �
'__inference_add_10_layer_call_fn_384997�d�a
Z�W
U�R
'�$
inputs/0���������(�
'�$
inputs/1���������(�
� "����������(��
B__inference_add_11_layer_call_and_return_conditional_losses_385140�d�a
Z�W
U�R
'�$
inputs/0���������(�
'�$
inputs/1���������(�
� "*�'
 �
0���������(�
� �
'__inference_add_11_layer_call_fn_385134�d�a
Z�W
U�R
'�$
inputs/0���������(�
'�$
inputs/1���������(�
� "����������(��
P__inference_average_pooling3d_30_layer_call_and_return_conditional_losses_385231�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
5__inference_average_pooling3d_30_layer_call_fn_385226�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
P__inference_average_pooling3d_31_layer_call_and_return_conditional_losses_385261�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
5__inference_average_pooling3d_31_layer_call_fn_385256�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
J__inference_concatenate_12_layer_call_and_return_conditional_losses_384812�d�a
Z�W
U�R
'�$
inputs/0���������(�
'�$
inputs/1���������(�
� "*�'
 �
0���������(�
� �
/__inference_concatenate_12_layer_call_fn_384805�d�a
Z�W
U�R
'�$
inputs/0���������(�
'�$
inputs/1���������(�
� "����������(��
D__inference_dense_10_layer_call_and_return_conditional_losses_385308h��4�1
*�'
%�"
inputs���������(�
� "*�'
 �
0���������(�
� �
)__inference_dense_10_layer_call_fn_385270[��4�1
*�'
%�"
inputs���������(�
� "����������(��
D__inference_dense_11_layer_call_and_return_conditional_losses_385201_��0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� 
)__inference_dense_11_layer_call_fn_385191R��0�-
&�#
!�
inputs����������
� "�����������
V__inference_global_average_pooling1d_5_layer_call_and_return_conditional_losses_385182{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
;__inference_global_average_pooling1d_5_layer_call_fn_385176nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
R__inference_layer_normalization_15_layer_call_and_return_conditional_losses_384864fDE4�1
*�'
%�"
inputs���������(�
� "*�'
 �
0���������(�
� �
7__inference_layer_normalization_15_layer_call_fn_384842YDE4�1
*�'
%�"
inputs���������(�
� "����������(��
R__inference_layer_normalization_16_layer_call_and_return_conditional_losses_385034f_`4�1
*�'
%�"
inputs���������(�
� "*�'
 �
0���������(�
� �
7__inference_layer_normalization_16_layer_call_fn_385012Y_`4�1
*�'
%�"
inputs���������(�
� "����������(��
R__inference_layer_normalization_17_layer_call_and_return_conditional_losses_385171fuv4�1
*�'
%�"
inputs���������(�
� "*�'
 �
0���������(�
� �
7__inference_layer_normalization_17_layer_call_fn_385149Yuv4�1
*�'
%�"
inputs���������(�
� "����������(��
L__inference_max_pooling3d_60_layer_call_and_return_conditional_losses_385211�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
1__inference_max_pooling3d_60_layer_call_fn_385206�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
L__inference_max_pooling3d_61_layer_call_and_return_conditional_losses_385221�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
1__inference_max_pooling3d_61_layer_call_fn_385216�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
L__inference_max_pooling3d_62_layer_call_and_return_conditional_losses_385241�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
1__inference_max_pooling3d_62_layer_call_fn_385236�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
L__inference_max_pooling3d_63_layer_call_and_return_conditional_losses_385251�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
1__inference_max_pooling3d_63_layer_call_fn_385246�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
C__inference_model_5_layer_call_and_return_conditional_losses_383953�:��������������DE��������_`��uv��E�B
;�8
.�+
input_16���������

p 

 
� "%�"
�
0���������
� �
C__inference_model_5_layer_call_and_return_conditional_losses_384043�:��������������DE��������_`��uv��E�B
;�8
.�+
input_16���������

p

 
� "%�"
�
0���������
� �
C__inference_model_5_layer_call_and_return_conditional_losses_384401�:��������������DE��������_`��uv��C�@
9�6
,�)
inputs���������

p 

 
� "%�"
�
0���������
� �
C__inference_model_5_layer_call_and_return_conditional_losses_384622�:��������������DE��������_`��uv��C�@
9�6
,�)
inputs���������

p

 
� "%�"
�
0���������
� �
(__inference_model_5_layer_call_fn_383367�:��������������DE��������_`��uv��E�B
;�8
.�+
input_16���������

p 

 
� "�����������
(__inference_model_5_layer_call_fn_383863�:��������������DE��������_`��uv��E�B
;�8
.�+
input_16���������

p

 
� "�����������
(__inference_model_5_layer_call_fn_384118�:��������������DE��������_`��uv��C�@
9�6
,�)
inputs���������

p 

 
� "�����������
(__inference_model_5_layer_call_fn_384187�:��������������DE��������_`��uv��C�@
9�6
,�)
inputs���������

p

 
� "�����������
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_384948���������i�f
_�\
$�!
query���������(�
$�!
value���������(�

 

 
p
p 
� "X�U
N�K
"�
0/0���������(�
%�"
0/1���������((
� �
R__inference_multi_head_attention_5_layer_call_and_return_conditional_losses_384991���������i�f
_�\
$�!
query���������(�
$�!
value���������(�

 

 
p
p
� "X�U
N�K
"�
0/0���������(�
%�"
0/1���������((
� �
7__inference_multi_head_attention_5_layer_call_fn_384888���������i�f
_�\
$�!
query���������(�
$�!
value���������(�

 

 
p
p 
� "J�G
 �
0���������(�
#� 
1���������((�
7__inference_multi_head_attention_5_layer_call_fn_384912���������i�f
_�\
$�!
query���������(�
$�!
value���������(�

 

 
p
p
� "J�G
 �
0���������(�
#� 
1���������((�
Q__inference_positional_encoder_15_layer_call_and_return_conditional_losses_384833p��<�9
2�/
-�*
encoded_tokens���������(�
� "*�'
 �
0���������(�
� �
6__inference_positional_encoder_15_layer_call_fn_384821c��<�9
2�/
-�*
encoded_tokens���������(�
� "����������(��
H__inference_sequential_5_layer_call_and_return_conditional_losses_382958x��D�A
:�7
-�*
dense_10_input���������(�
p 

 
� "*�'
 �
0���������(�
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_382967x��D�A
:�7
-�*
dense_10_input���������(�
p

 
� "*�'
 �
0���������(�
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_385090p��<�9
2�/
%�"
inputs���������(�
p 

 
� "*�'
 �
0���������(�
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_385128p��<�9
2�/
%�"
inputs���������(�
p

 
� "*�'
 �
0���������(�
� �
-__inference_sequential_5_layer_call_fn_382903k��D�A
:�7
-�*
dense_10_input���������(�
p 

 
� "����������(��
-__inference_sequential_5_layer_call_fn_382949k��D�A
:�7
-�*
dense_10_input���������(�
p

 
� "����������(��
-__inference_sequential_5_layer_call_fn_385043c��<�9
2�/
%�"
inputs���������(�
p 

 
� "����������(��
-__inference_sequential_5_layer_call_fn_385052c��<�9
2�/
%�"
inputs���������(�
p

 
� "����������(��
$__inference_signature_wrapper_384693�:��������������DE��������_`��uv��I�F
� 
?�<
:
input_16.�+
input_16���������
"3�0
.
dense_11"�
dense_11����������
P__inference_tubelet_embedding_30_layer_call_and_return_conditional_losses_384746w������;�8
1�.
,�)
videos���������

� "*�'
 �
0���������(�
� �
5__inference_tubelet_embedding_30_layer_call_fn_384710j������;�8
1�.
,�)
videos���������

� "����������(��
P__inference_tubelet_embedding_31_layer_call_and_return_conditional_losses_384799w������;�8
1�.
,�)
videos���������

� "*�'
 �
0���������(�
� �
5__inference_tubelet_embedding_31_layer_call_fn_384763j������;�8
1�.
,�)
videos���������

� "����������(�