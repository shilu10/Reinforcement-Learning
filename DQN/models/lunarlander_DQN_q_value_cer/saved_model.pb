сё
ѕФ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58цы
Њ
)Adam/v/deep_q_network2d_38/dense_116/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/v/deep_q_network2d_38/dense_116/bias
Ѓ
=Adam/v/deep_q_network2d_38/dense_116/bias/Read/ReadVariableOpReadVariableOp)Adam/v/deep_q_network2d_38/dense_116/bias*
_output_shapes
:*
dtype0
Њ
)Adam/m/deep_q_network2d_38/dense_116/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/m/deep_q_network2d_38/dense_116/bias
Ѓ
=Adam/m/deep_q_network2d_38/dense_116/bias/Read/ReadVariableOpReadVariableOp)Adam/m/deep_q_network2d_38/dense_116/bias*
_output_shapes
:*
dtype0
В
+Adam/v/deep_q_network2d_38/dense_116/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*<
shared_name-+Adam/v/deep_q_network2d_38/dense_116/kernel
Ћ
?Adam/v/deep_q_network2d_38/dense_116/kernel/Read/ReadVariableOpReadVariableOp+Adam/v/deep_q_network2d_38/dense_116/kernel*
_output_shapes

:@*
dtype0
В
+Adam/m/deep_q_network2d_38/dense_116/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*<
shared_name-+Adam/m/deep_q_network2d_38/dense_116/kernel
Ћ
?Adam/m/deep_q_network2d_38/dense_116/kernel/Read/ReadVariableOpReadVariableOp+Adam/m/deep_q_network2d_38/dense_116/kernel*
_output_shapes

:@*
dtype0
Њ
)Adam/v/deep_q_network2d_38/dense_115/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/v/deep_q_network2d_38/dense_115/bias
Ѓ
=Adam/v/deep_q_network2d_38/dense_115/bias/Read/ReadVariableOpReadVariableOp)Adam/v/deep_q_network2d_38/dense_115/bias*
_output_shapes
:@*
dtype0
Њ
)Adam/m/deep_q_network2d_38/dense_115/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/m/deep_q_network2d_38/dense_115/bias
Ѓ
=Adam/m/deep_q_network2d_38/dense_115/bias/Read/ReadVariableOpReadVariableOp)Adam/m/deep_q_network2d_38/dense_115/bias*
_output_shapes
:@*
dtype0
В
+Adam/v/deep_q_network2d_38/dense_115/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*<
shared_name-+Adam/v/deep_q_network2d_38/dense_115/kernel
Ћ
?Adam/v/deep_q_network2d_38/dense_115/kernel/Read/ReadVariableOpReadVariableOp+Adam/v/deep_q_network2d_38/dense_115/kernel*
_output_shapes

:@@*
dtype0
В
+Adam/m/deep_q_network2d_38/dense_115/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*<
shared_name-+Adam/m/deep_q_network2d_38/dense_115/kernel
Ћ
?Adam/m/deep_q_network2d_38/dense_115/kernel/Read/ReadVariableOpReadVariableOp+Adam/m/deep_q_network2d_38/dense_115/kernel*
_output_shapes

:@@*
dtype0
Њ
)Adam/v/deep_q_network2d_38/dense_114/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/v/deep_q_network2d_38/dense_114/bias
Ѓ
=Adam/v/deep_q_network2d_38/dense_114/bias/Read/ReadVariableOpReadVariableOp)Adam/v/deep_q_network2d_38/dense_114/bias*
_output_shapes
:@*
dtype0
Њ
)Adam/m/deep_q_network2d_38/dense_114/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/m/deep_q_network2d_38/dense_114/bias
Ѓ
=Adam/m/deep_q_network2d_38/dense_114/bias/Read/ReadVariableOpReadVariableOp)Adam/m/deep_q_network2d_38/dense_114/bias*
_output_shapes
:@*
dtype0
В
+Adam/v/deep_q_network2d_38/dense_114/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*<
shared_name-+Adam/v/deep_q_network2d_38/dense_114/kernel
Ћ
?Adam/v/deep_q_network2d_38/dense_114/kernel/Read/ReadVariableOpReadVariableOp+Adam/v/deep_q_network2d_38/dense_114/kernel*
_output_shapes

:@*
dtype0
В
+Adam/m/deep_q_network2d_38/dense_114/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*<
shared_name-+Adam/m/deep_q_network2d_38/dense_114/kernel
Ћ
?Adam/m/deep_q_network2d_38/dense_114/kernel/Read/ReadVariableOpReadVariableOp+Adam/m/deep_q_network2d_38/dense_114/kernel*
_output_shapes

:@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

"deep_q_network2d_38/dense_116/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"deep_q_network2d_38/dense_116/bias

6deep_q_network2d_38/dense_116/bias/Read/ReadVariableOpReadVariableOp"deep_q_network2d_38/dense_116/bias*
_output_shapes
:*
dtype0
Є
$deep_q_network2d_38/dense_116/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*5
shared_name&$deep_q_network2d_38/dense_116/kernel

8deep_q_network2d_38/dense_116/kernel/Read/ReadVariableOpReadVariableOp$deep_q_network2d_38/dense_116/kernel*
_output_shapes

:@*
dtype0

"deep_q_network2d_38/dense_115/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"deep_q_network2d_38/dense_115/bias

6deep_q_network2d_38/dense_115/bias/Read/ReadVariableOpReadVariableOp"deep_q_network2d_38/dense_115/bias*
_output_shapes
:@*
dtype0
Є
$deep_q_network2d_38/dense_115/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*5
shared_name&$deep_q_network2d_38/dense_115/kernel

8deep_q_network2d_38/dense_115/kernel/Read/ReadVariableOpReadVariableOp$deep_q_network2d_38/dense_115/kernel*
_output_shapes

:@@*
dtype0

"deep_q_network2d_38/dense_114/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"deep_q_network2d_38/dense_114/bias

6deep_q_network2d_38/dense_114/bias/Read/ReadVariableOpReadVariableOp"deep_q_network2d_38/dense_114/bias*
_output_shapes
:@*
dtype0
Є
$deep_q_network2d_38/dense_114/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*5
shared_name&$deep_q_network2d_38/dense_114/kernel

8deep_q_network2d_38/dense_114/kernel/Read/ReadVariableOpReadVariableOp$deep_q_network2d_38/dense_114/kernel*
_output_shapes

:@*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Ѓ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1$deep_q_network2d_38/dense_114/kernel"deep_q_network2d_38/dense_114/bias$deep_q_network2d_38/dense_115/kernel"deep_q_network2d_38/dense_115/bias$deep_q_network2d_38/dense_116/kernel"deep_q_network2d_38/dense_116/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 */
f*R(
&__inference_signature_wrapper_32048758

NoOpNoOp
з(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*(
value(B( Bў'
ё
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

fc3
	optimizer
loss

signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
А
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
І
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias*
І
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

kernel
bias*
І
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias*

/
_variables
0_iterations
1_learning_rate
2_index_dict
3
_momentums
4_velocities
5_update_step_xla*
* 

6serving_default* 
d^
VARIABLE_VALUE$deep_q_network2d_38/dense_114/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"deep_q_network2d_38/dense_114/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$deep_q_network2d_38/dense_115/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"deep_q_network2d_38/dense_115/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$deep_q_network2d_38/dense_116/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"deep_q_network2d_38/dense_116/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1

2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

<trace_0* 

=trace_0* 

0
1*

0
1*
* 

>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 

0
1*

0
1*
* 

Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 
b
00
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
L0
N1
P2
R3
T4
V5*
.
M0
O1
Q2
S3
U4
W5*
P
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3
\trace_4
]trace_5* 
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
vp
VARIABLE_VALUE+Adam/m/deep_q_network2d_38/dense_114/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE+Adam/v/deep_q_network2d_38/dense_114/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/m/deep_q_network2d_38/dense_114/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/v/deep_q_network2d_38/dense_114/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE+Adam/m/deep_q_network2d_38/dense_115/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE+Adam/v/deep_q_network2d_38/dense_115/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/m/deep_q_network2d_38/dense_115/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/v/deep_q_network2d_38/dense_115/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE+Adam/m/deep_q_network2d_38/dense_116/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/v/deep_q_network2d_38/dense_116/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/m/deep_q_network2d_38/dense_116/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/deep_q_network2d_38/dense_116/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ю
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename8deep_q_network2d_38/dense_114/kernel/Read/ReadVariableOp6deep_q_network2d_38/dense_114/bias/Read/ReadVariableOp8deep_q_network2d_38/dense_115/kernel/Read/ReadVariableOp6deep_q_network2d_38/dense_115/bias/Read/ReadVariableOp8deep_q_network2d_38/dense_116/kernel/Read/ReadVariableOp6deep_q_network2d_38/dense_116/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp?Adam/m/deep_q_network2d_38/dense_114/kernel/Read/ReadVariableOp?Adam/v/deep_q_network2d_38/dense_114/kernel/Read/ReadVariableOp=Adam/m/deep_q_network2d_38/dense_114/bias/Read/ReadVariableOp=Adam/v/deep_q_network2d_38/dense_114/bias/Read/ReadVariableOp?Adam/m/deep_q_network2d_38/dense_115/kernel/Read/ReadVariableOp?Adam/v/deep_q_network2d_38/dense_115/kernel/Read/ReadVariableOp=Adam/m/deep_q_network2d_38/dense_115/bias/Read/ReadVariableOp=Adam/v/deep_q_network2d_38/dense_115/bias/Read/ReadVariableOp?Adam/m/deep_q_network2d_38/dense_116/kernel/Read/ReadVariableOp?Adam/v/deep_q_network2d_38/dense_116/kernel/Read/ReadVariableOp=Adam/m/deep_q_network2d_38/dense_116/bias/Read/ReadVariableOp=Adam/v/deep_q_network2d_38/dense_116/bias/Read/ReadVariableOpConst*!
Tin
2	*
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
GPU2 *0J 8 **
f%R#
!__inference__traced_save_32048941
Й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename$deep_q_network2d_38/dense_114/kernel"deep_q_network2d_38/dense_114/bias$deep_q_network2d_38/dense_115/kernel"deep_q_network2d_38/dense_115/bias$deep_q_network2d_38/dense_116/kernel"deep_q_network2d_38/dense_116/bias	iterationlearning_rate+Adam/m/deep_q_network2d_38/dense_114/kernel+Adam/v/deep_q_network2d_38/dense_114/kernel)Adam/m/deep_q_network2d_38/dense_114/bias)Adam/v/deep_q_network2d_38/dense_114/bias+Adam/m/deep_q_network2d_38/dense_115/kernel+Adam/v/deep_q_network2d_38/dense_115/kernel)Adam/m/deep_q_network2d_38/dense_115/bias)Adam/v/deep_q_network2d_38/dense_115/bias+Adam/m/deep_q_network2d_38/dense_116/kernel+Adam/v/deep_q_network2d_38/dense_116/kernel)Adam/m/deep_q_network2d_38/dense_116/bias)Adam/v/deep_q_network2d_38/dense_116/bias* 
Tin
2*
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
GPU2 *0J 8 *-
f(R&
$__inference__traced_restore_32049011і
в5
т
!__inference__traced_save_32048941
file_prefixC
?savev2_deep_q_network2d_38_dense_114_kernel_read_readvariableopA
=savev2_deep_q_network2d_38_dense_114_bias_read_readvariableopC
?savev2_deep_q_network2d_38_dense_115_kernel_read_readvariableopA
=savev2_deep_q_network2d_38_dense_115_bias_read_readvariableopC
?savev2_deep_q_network2d_38_dense_116_kernel_read_readvariableopA
=savev2_deep_q_network2d_38_dense_116_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopJ
Fsavev2_adam_m_deep_q_network2d_38_dense_114_kernel_read_readvariableopJ
Fsavev2_adam_v_deep_q_network2d_38_dense_114_kernel_read_readvariableopH
Dsavev2_adam_m_deep_q_network2d_38_dense_114_bias_read_readvariableopH
Dsavev2_adam_v_deep_q_network2d_38_dense_114_bias_read_readvariableopJ
Fsavev2_adam_m_deep_q_network2d_38_dense_115_kernel_read_readvariableopJ
Fsavev2_adam_v_deep_q_network2d_38_dense_115_kernel_read_readvariableopH
Dsavev2_adam_m_deep_q_network2d_38_dense_115_bias_read_readvariableopH
Dsavev2_adam_v_deep_q_network2d_38_dense_115_bias_read_readvariableopJ
Fsavev2_adam_m_deep_q_network2d_38_dense_116_kernel_read_readvariableopJ
Fsavev2_adam_v_deep_q_network2d_38_dense_116_kernel_read_readvariableopH
Dsavev2_adam_m_deep_q_network2d_38_dense_116_bias_read_readvariableopH
Dsavev2_adam_v_deep_q_network2d_38_dense_116_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ш
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ё
valueчBфB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0?savev2_deep_q_network2d_38_dense_114_kernel_read_readvariableop=savev2_deep_q_network2d_38_dense_114_bias_read_readvariableop?savev2_deep_q_network2d_38_dense_115_kernel_read_readvariableop=savev2_deep_q_network2d_38_dense_115_bias_read_readvariableop?savev2_deep_q_network2d_38_dense_116_kernel_read_readvariableop=savev2_deep_q_network2d_38_dense_116_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopFsavev2_adam_m_deep_q_network2d_38_dense_114_kernel_read_readvariableopFsavev2_adam_v_deep_q_network2d_38_dense_114_kernel_read_readvariableopDsavev2_adam_m_deep_q_network2d_38_dense_114_bias_read_readvariableopDsavev2_adam_v_deep_q_network2d_38_dense_114_bias_read_readvariableopFsavev2_adam_m_deep_q_network2d_38_dense_115_kernel_read_readvariableopFsavev2_adam_v_deep_q_network2d_38_dense_115_kernel_read_readvariableopDsavev2_adam_m_deep_q_network2d_38_dense_115_bias_read_readvariableopDsavev2_adam_v_deep_q_network2d_38_dense_115_bias_read_readvariableopFsavev2_adam_m_deep_q_network2d_38_dense_116_kernel_read_readvariableopFsavev2_adam_v_deep_q_network2d_38_dense_116_kernel_read_readvariableopDsavev2_adam_m_deep_q_network2d_38_dense_116_bias_read_readvariableopDsavev2_adam_v_deep_q_network2d_38_dense_116_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *#
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*­
_input_shapes
: :@:@:@@:@:@:: : :@:@:@:@:@@:@@:@:@:@:@::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$	 

_output_shapes

:@:$
 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
Э

,__inference_dense_114_layer_call_fn_32048808

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_114_layer_call_and_return_conditional_losses_32048616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ј
G__inference_dense_114_layer_call_and_return_conditional_losses_32048616

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ј
G__inference_dense_115_layer_call_and_return_conditional_losses_32048839

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
И
O
#__inference__update_step_xla_185215
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@: *
	_noinline(:H D

_output_shapes

:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Э

,__inference_dense_115_layer_call_fn_32048828

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_115_layer_call_and_return_conditional_losses_32048633o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ъ	
ј
G__inference_dense_116_layer_call_and_return_conditional_losses_32048858

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
р%
с
#__inference__wrapped_model_32048598
input_1N
<deep_q_network2d_38_dense_114_matmul_readvariableop_resource:@K
=deep_q_network2d_38_dense_114_biasadd_readvariableop_resource:@N
<deep_q_network2d_38_dense_115_matmul_readvariableop_resource:@@K
=deep_q_network2d_38_dense_115_biasadd_readvariableop_resource:@N
<deep_q_network2d_38_dense_116_matmul_readvariableop_resource:@K
=deep_q_network2d_38_dense_116_biasadd_readvariableop_resource:
identityЂ4deep_q_network2d_38/dense_114/BiasAdd/ReadVariableOpЂ3deep_q_network2d_38/dense_114/MatMul/ReadVariableOpЂ4deep_q_network2d_38/dense_115/BiasAdd/ReadVariableOpЂ3deep_q_network2d_38/dense_115/MatMul/ReadVariableOpЂ4deep_q_network2d_38/dense_116/BiasAdd/ReadVariableOpЂ3deep_q_network2d_38/dense_116/MatMul/ReadVariableOpА
3deep_q_network2d_38/dense_114/MatMul/ReadVariableOpReadVariableOp<deep_q_network2d_38_dense_114_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0І
$deep_q_network2d_38/dense_114/MatMulMatMulinput_1;deep_q_network2d_38/dense_114/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ў
4deep_q_network2d_38/dense_114/BiasAdd/ReadVariableOpReadVariableOp=deep_q_network2d_38_dense_114_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0а
%deep_q_network2d_38/dense_114/BiasAddBiasAdd.deep_q_network2d_38/dense_114/MatMul:product:0<deep_q_network2d_38/dense_114/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
"deep_q_network2d_38/dense_114/ReluRelu.deep_q_network2d_38/dense_114/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@А
3deep_q_network2d_38/dense_115/MatMul/ReadVariableOpReadVariableOp<deep_q_network2d_38_dense_115_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Я
$deep_q_network2d_38/dense_115/MatMulMatMul0deep_q_network2d_38/dense_114/Relu:activations:0;deep_q_network2d_38/dense_115/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ў
4deep_q_network2d_38/dense_115/BiasAdd/ReadVariableOpReadVariableOp=deep_q_network2d_38_dense_115_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0а
%deep_q_network2d_38/dense_115/BiasAddBiasAdd.deep_q_network2d_38/dense_115/MatMul:product:0<deep_q_network2d_38/dense_115/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
"deep_q_network2d_38/dense_115/ReluRelu.deep_q_network2d_38/dense_115/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@А
3deep_q_network2d_38/dense_116/MatMul/ReadVariableOpReadVariableOp<deep_q_network2d_38_dense_116_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
$deep_q_network2d_38/dense_116/MatMulMatMul0deep_q_network2d_38/dense_115/Relu:activations:0;deep_q_network2d_38/dense_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
4deep_q_network2d_38/dense_116/BiasAdd/ReadVariableOpReadVariableOp=deep_q_network2d_38_dense_116_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0а
%deep_q_network2d_38/dense_116/BiasAddBiasAdd.deep_q_network2d_38/dense_116/MatMul:product:0<deep_q_network2d_38/dense_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ}
IdentityIdentity.deep_q_network2d_38/dense_116/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp5^deep_q_network2d_38/dense_114/BiasAdd/ReadVariableOp4^deep_q_network2d_38/dense_114/MatMul/ReadVariableOp5^deep_q_network2d_38/dense_115/BiasAdd/ReadVariableOp4^deep_q_network2d_38/dense_115/MatMul/ReadVariableOp5^deep_q_network2d_38/dense_116/BiasAdd/ReadVariableOp4^deep_q_network2d_38/dense_116/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2l
4deep_q_network2d_38/dense_114/BiasAdd/ReadVariableOp4deep_q_network2d_38/dense_114/BiasAdd/ReadVariableOp2j
3deep_q_network2d_38/dense_114/MatMul/ReadVariableOp3deep_q_network2d_38/dense_114/MatMul/ReadVariableOp2l
4deep_q_network2d_38/dense_115/BiasAdd/ReadVariableOp4deep_q_network2d_38/dense_115/BiasAdd/ReadVariableOp2j
3deep_q_network2d_38/dense_115/MatMul/ReadVariableOp3deep_q_network2d_38/dense_115/MatMul/ReadVariableOp2l
4deep_q_network2d_38/dense_116/BiasAdd/ReadVariableOp4deep_q_network2d_38/dense_116/BiasAdd/ReadVariableOp2j
3deep_q_network2d_38/dense_116/MatMul/ReadVariableOp3deep_q_network2d_38/dense_116/MatMul/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
	

6__inference_deep_q_network2d_38_layer_call_fn_32048775	
state
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048656o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_namestate
И
O
#__inference__update_step_xla_185195
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@: *
	_noinline(:H D

_output_shapes

:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ќ
K
#__inference__update_step_xla_185200
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable


ј
G__inference_dense_115_layer_call_and_return_conditional_losses_32048633

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ќ
K
#__inference__update_step_xla_185220
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
о
Е
Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048737
input_1$
dense_114_32048721:@ 
dense_114_32048723:@$
dense_115_32048726:@@ 
dense_115_32048728:@$
dense_116_32048731:@ 
dense_116_32048733:
identityЂ!dense_114/StatefulPartitionedCallЂ!dense_115/StatefulPartitionedCallЂ!dense_116/StatefulPartitionedCall
!dense_114/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_114_32048721dense_114_32048723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_114_layer_call_and_return_conditional_losses_32048616Ѓ
!dense_115/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0dense_115_32048726dense_115_32048728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_115_layer_call_and_return_conditional_losses_32048633Ѓ
!dense_116/StatefulPartitionedCallStatefulPartitionedCall*dense_115/StatefulPartitionedCall:output:0dense_116_32048731dense_116_32048733*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_116_layer_call_and_return_conditional_losses_32048649y
IdentityIdentity*dense_116/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџВ
NoOpNoOp"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall"^dense_116/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall2F
!dense_116/StatefulPartitionedCall!dense_116/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1


ј
G__inference_dense_114_layer_call_and_return_conditional_losses_32048819

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048799	
state:
(dense_114_matmul_readvariableop_resource:@7
)dense_114_biasadd_readvariableop_resource:@:
(dense_115_matmul_readvariableop_resource:@@7
)dense_115_biasadd_readvariableop_resource:@:
(dense_116_matmul_readvariableop_resource:@7
)dense_116_biasadd_readvariableop_resource:
identityЂ dense_114/BiasAdd/ReadVariableOpЂdense_114/MatMul/ReadVariableOpЂ dense_115/BiasAdd/ReadVariableOpЂdense_115/MatMul/ReadVariableOpЂ dense_116/BiasAdd/ReadVariableOpЂdense_116/MatMul/ReadVariableOp
dense_114/MatMul/ReadVariableOpReadVariableOp(dense_114_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0|
dense_114/MatMulMatMulstate'dense_114/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
 dense_114/BiasAdd/ReadVariableOpReadVariableOp)dense_114_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_114/BiasAddBiasAdddense_114/MatMul:product:0(dense_114/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@d
dense_114/ReluReludense_114/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_115/MatMul/ReadVariableOpReadVariableOp(dense_115_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
dense_115/MatMulMatMuldense_114/Relu:activations:0'dense_115/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
 dense_115/BiasAdd/ReadVariableOpReadVariableOp)dense_115_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_115/BiasAddBiasAdddense_115/MatMul:product:0(dense_115/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@d
dense_115/ReluReludense_115/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_116/MatMul/ReadVariableOpReadVariableOp(dense_116_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_116/MatMulMatMuldense_115/Relu:activations:0'dense_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_116/BiasAdd/ReadVariableOpReadVariableOp)dense_116_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_116/BiasAddBiasAdddense_116/MatMul:product:0(dense_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentitydense_116/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_114/BiasAdd/ReadVariableOp ^dense_114/MatMul/ReadVariableOp!^dense_115/BiasAdd/ReadVariableOp ^dense_115/MatMul/ReadVariableOp!^dense_116/BiasAdd/ReadVariableOp ^dense_116/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2D
 dense_114/BiasAdd/ReadVariableOp dense_114/BiasAdd/ReadVariableOp2B
dense_114/MatMul/ReadVariableOpdense_114/MatMul/ReadVariableOp2D
 dense_115/BiasAdd/ReadVariableOp dense_115/BiasAdd/ReadVariableOp2B
dense_115/MatMul/ReadVariableOpdense_115/MatMul/ReadVariableOp2D
 dense_116/BiasAdd/ReadVariableOp dense_116/BiasAdd/ReadVariableOp2B
dense_116/MatMul/ReadVariableOpdense_116/MatMul/ReadVariableOp:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_namestate
Ќ
K
#__inference__update_step_xla_185210
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
И
O
#__inference__update_step_xla_185205
gradient
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@@: *
	_noinline(:H D

_output_shapes

:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ю

&__inference_signature_wrapper_32048758
input_1
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__wrapped_model_32048598o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Э

,__inference_dense_116_layer_call_fn_32048848

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_116_layer_call_and_return_conditional_losses_32048649o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ъ	
ј
G__inference_dense_116_layer_call_and_return_conditional_losses_32048649

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
п\
З
$__inference__traced_restore_32049011
file_prefixG
5assignvariableop_deep_q_network2d_38_dense_114_kernel:@C
5assignvariableop_1_deep_q_network2d_38_dense_114_bias:@I
7assignvariableop_2_deep_q_network2d_38_dense_115_kernel:@@C
5assignvariableop_3_deep_q_network2d_38_dense_115_bias:@I
7assignvariableop_4_deep_q_network2d_38_dense_116_kernel:@C
5assignvariableop_5_deep_q_network2d_38_dense_116_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: P
>assignvariableop_8_adam_m_deep_q_network2d_38_dense_114_kernel:@P
>assignvariableop_9_adam_v_deep_q_network2d_38_dense_114_kernel:@K
=assignvariableop_10_adam_m_deep_q_network2d_38_dense_114_bias:@K
=assignvariableop_11_adam_v_deep_q_network2d_38_dense_114_bias:@Q
?assignvariableop_12_adam_m_deep_q_network2d_38_dense_115_kernel:@@Q
?assignvariableop_13_adam_v_deep_q_network2d_38_dense_115_kernel:@@K
=assignvariableop_14_adam_m_deep_q_network2d_38_dense_115_bias:@K
=assignvariableop_15_adam_v_deep_q_network2d_38_dense_115_bias:@Q
?assignvariableop_16_adam_m_deep_q_network2d_38_dense_116_kernel:@Q
?assignvariableop_17_adam_v_deep_q_network2d_38_dense_116_kernel:@K
=assignvariableop_18_adam_m_deep_q_network2d_38_dense_116_bias:K
=assignvariableop_19_adam_v_deep_q_network2d_38_dense_116_bias:
identity_21ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ы
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ё
valueчBфB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOpAssignVariableOp5assignvariableop_deep_q_network2d_38_dense_114_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_1AssignVariableOp5assignvariableop_1_deep_q_network2d_38_dense_114_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp7assignvariableop_2_deep_q_network2d_38_dense_115_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_3AssignVariableOp5assignvariableop_3_deep_q_network2d_38_dense_115_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_4AssignVariableOp7assignvariableop_4_deep_q_network2d_38_dense_116_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_5AssignVariableOp5assignvariableop_5_deep_q_network2d_38_dense_116_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_8AssignVariableOp>assignvariableop_8_adam_m_deep_q_network2d_38_dense_114_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_9AssignVariableOp>assignvariableop_9_adam_v_deep_q_network2d_38_dense_114_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_10AssignVariableOp=assignvariableop_10_adam_m_deep_q_network2d_38_dense_114_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_11AssignVariableOp=assignvariableop_11_adam_v_deep_q_network2d_38_dense_114_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_12AssignVariableOp?assignvariableop_12_adam_m_deep_q_network2d_38_dense_115_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_13AssignVariableOp?assignvariableop_13_adam_v_deep_q_network2d_38_dense_115_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_14AssignVariableOp=assignvariableop_14_adam_m_deep_q_network2d_38_dense_115_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_15AssignVariableOp=assignvariableop_15_adam_v_deep_q_network2d_38_dense_115_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_16AssignVariableOp?assignvariableop_16_adam_m_deep_q_network2d_38_dense_116_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_17AssignVariableOp?assignvariableop_17_adam_v_deep_q_network2d_38_dense_116_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_18AssignVariableOp=assignvariableop_18_adam_m_deep_q_network2d_38_dense_116_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_19AssignVariableOp=assignvariableop_19_adam_v_deep_q_network2d_38_dense_116_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: є
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
и
Г
Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048656	
state$
dense_114_32048617:@ 
dense_114_32048619:@$
dense_115_32048634:@@ 
dense_115_32048636:@$
dense_116_32048650:@ 
dense_116_32048652:
identityЂ!dense_114/StatefulPartitionedCallЂ!dense_115/StatefulPartitionedCallЂ!dense_116/StatefulPartitionedCallў
!dense_114/StatefulPartitionedCallStatefulPartitionedCallstatedense_114_32048617dense_114_32048619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_114_layer_call_and_return_conditional_losses_32048616Ѓ
!dense_115/StatefulPartitionedCallStatefulPartitionedCall*dense_114/StatefulPartitionedCall:output:0dense_115_32048634dense_115_32048636*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_115_layer_call_and_return_conditional_losses_32048633Ѓ
!dense_116/StatefulPartitionedCallStatefulPartitionedCall*dense_115/StatefulPartitionedCall:output:0dense_116_32048650dense_116_32048652*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_dense_116_layer_call_and_return_conditional_losses_32048649y
IdentityIdentity*dense_116/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџВ
NoOpNoOp"^dense_114/StatefulPartitionedCall"^dense_115/StatefulPartitionedCall"^dense_116/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2F
!dense_114/StatefulPartitionedCall!dense_114/StatefulPartitionedCall2F
!dense_115/StatefulPartitionedCall!dense_115/StatefulPartitionedCall2F
!dense_116/StatefulPartitionedCall!dense_116/StatefulPartitionedCall:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_namestate
	

6__inference_deep_q_network2d_38_layer_call_fn_32048671
input_1
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048656o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ћ
serving_default
;
input_10
serving_default_input_1:0џџџџџџџџџ<
output_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:u

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

fc3
	optimizer
loss

signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ы
trace_0
trace_12
6__inference_deep_q_network2d_38_layer_call_fn_32048671
6__inference_deep_q_network2d_38_layer_call_fn_32048775Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12Ъ
Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048799
Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048737Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ЮBЫ
#__inference__wrapped_model_32048598input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Л
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer

/
_variables
0_iterations
1_learning_rate
2_index_dict
3
_momentums
4_velocities
5_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
6serving_default"
signature_map
6:4@2$deep_q_network2d_38/dense_114/kernel
0:.@2"deep_q_network2d_38/dense_114/bias
6:4@@2$deep_q_network2d_38/dense_115/kernel
0:.@2"deep_q_network2d_38/dense_115/bias
6:4@2$deep_q_network2d_38/dense_116/kernel
0:.2"deep_q_network2d_38/dense_116/bias
 "
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
6__inference_deep_q_network2d_38_layer_call_fn_32048671input_1"Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
шBх
6__inference_deep_q_network2d_38_layer_call_fn_32048775state"Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048799state"Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048737input_1"Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
№
<trace_02г
,__inference_dense_114_layer_call_fn_32048808Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z<trace_0

=trace_02ю
G__inference_dense_114_layer_call_and_return_conditional_losses_32048819Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z=trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
№
Ctrace_02г
,__inference_dense_115_layer_call_fn_32048828Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zCtrace_0

Dtrace_02ю
G__inference_dense_115_layer_call_and_return_conditional_losses_32048839Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zDtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
№
Jtrace_02г
,__inference_dense_116_layer_call_fn_32048848Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zJtrace_0

Ktrace_02ю
G__inference_dense_116_layer_call_and_return_conditional_losses_32048858Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zKtrace_0
~
00
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
L0
N1
P2
R3
T4
V5"
trackable_list_wrapper
J
M0
O1
Q2
S3
U4
W5"
trackable_list_wrapper
Й
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3
\trace_4
]trace_52
#__inference__update_step_xla_185195
#__inference__update_step_xla_185200
#__inference__update_step_xla_185205
#__inference__update_step_xla_185210
#__inference__update_step_xla_185215
#__inference__update_step_xla_185220Й
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zXtrace_0zYtrace_1zZtrace_2z[trace_3z\trace_4z]trace_5
ЭBЪ
&__inference_signature_wrapper_32048758input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
рBн
,__inference_dense_114_layer_call_fn_32048808inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_dense_114_layer_call_and_return_conditional_losses_32048819inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
рBн
,__inference_dense_115_layer_call_fn_32048828inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_dense_115_layer_call_and_return_conditional_losses_32048839inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
рBн
,__inference_dense_116_layer_call_fn_32048848inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_dense_116_layer_call_and_return_conditional_losses_32048858inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
;:9@2+Adam/m/deep_q_network2d_38/dense_114/kernel
;:9@2+Adam/v/deep_q_network2d_38/dense_114/kernel
5:3@2)Adam/m/deep_q_network2d_38/dense_114/bias
5:3@2)Adam/v/deep_q_network2d_38/dense_114/bias
;:9@@2+Adam/m/deep_q_network2d_38/dense_115/kernel
;:9@@2+Adam/v/deep_q_network2d_38/dense_115/kernel
5:3@2)Adam/m/deep_q_network2d_38/dense_115/bias
5:3@2)Adam/v/deep_q_network2d_38/dense_115/bias
;:9@2+Adam/m/deep_q_network2d_38/dense_116/kernel
;:9@2+Adam/v/deep_q_network2d_38/dense_116/kernel
5:32)Adam/m/deep_q_network2d_38/dense_116/bias
5:32)Adam/v/deep_q_network2d_38/dense_116/bias
јBѕ
#__inference__update_step_xla_185195gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
#__inference__update_step_xla_185200gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
#__inference__update_step_xla_185205gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
#__inference__update_step_xla_185210gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
#__inference__update_step_xla_185215gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
#__inference__update_step_xla_185220gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
#__inference__update_step_xla_185195nhЂe
^Ђ[

gradient@
41	Ђ
њ@

p
` VariableSpec 
`рсў№Др?
Њ "
 
#__inference__update_step_xla_185200f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`рфў№Др?
Њ "
 
#__inference__update_step_xla_185205nhЂe
^Ђ[

gradient@@
41	Ђ
њ@@

p
` VariableSpec 
`рЫ№Др?
Њ "
 
#__inference__update_step_xla_185210f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
` Ы№Др?
Њ "
 
#__inference__update_step_xla_185215nhЂe
^Ђ[

gradient@
41	Ђ
њ@

p
` VariableSpec 
`рляДр?
Њ "
 
#__inference__update_step_xla_185220f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` ляДр?
Њ "
 
#__inference__wrapped_model_32048598o0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџН
Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048737h0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Л
Q__inference_deep_q_network2d_38_layer_call_and_return_conditional_losses_32048799f.Ђ+
$Ђ!

stateџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
6__inference_deep_q_network2d_38_layer_call_fn_32048671]0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
6__inference_deep_q_network2d_38_layer_call_fn_32048775[.Ђ+
$Ђ!

stateџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЎ
G__inference_dense_114_layer_call_and_return_conditional_losses_32048819c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
,__inference_dense_114_layer_call_fn_32048808X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ@Ў
G__inference_dense_115_layer_call_and_return_conditional_losses_32048839c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
,__inference_dense_115_layer_call_fn_32048828X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Ў
G__inference_dense_116_layer_call_and_return_conditional_losses_32048858c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
,__inference_dense_116_layer_call_fn_32048848X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџЄ
&__inference_signature_wrapper_32048758z;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"3Њ0
.
output_1"
output_1џџџџџџџџџ