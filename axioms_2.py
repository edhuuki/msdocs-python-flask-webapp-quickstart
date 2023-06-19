# Author: Erik Huuki
# The backend software of EE403 capstone Design project
# Description:
# Axioms_2 is a symbolic math library that converts mathematical statements binary trees
# This is for the purposes of automating symbolic math operations that 
# undergraduates and professional engineers frequent
import numpy as np


class expr:
    def __init__(self,exp:str=None,root=None,**kwargs):
        # str expressions and roots of trees to define expression objects
        self.root = root

        if isinstance(exp,str):
            self.root = self.exp2tree(exp)
        
        self.dir = {}
        self.map()

    def exp2tree(self,exp):
        # Converts the raw string into a tree
        # ex. list2tree('a+b')=> node(val='+', left=node('a'), right=node('b'))

        # check for mismatched delimiters
        if exp.count('(')!= exp.count(')'):
            raise Exception('Mismatched delimiters')
        
        if '-^' in exp:
            raise Exception('Cannot determine precendence of operators')
        
        exp = exp.replace(' ','')       # Remove blank space

        # Need to check for incomplete expressions / errors in expressions

        # tokenizes the string expression into a list
        exp_list = _tokenize(exp)
        protected_string = ['(',')','==','=','<','>','<=','>=','+','-','*','/','^','&','|',',']

        # replaces recognized data with constants and throws errors for strings that cannot be interpreted
        exp_list = [val if val in protected_string else self._str2values(val) for val in exp_list]  
        
        return self.list2tree(exp_list) # returns thee root of a tree generated from the exp list

    def _str2values(self,s):
        # Converts a string to values that are recognized as either bool, int, float, complex
        # Where naming conventions match python 

        special_cases = {   # special strings that to return as constants
            '':None,
            'True':True,
            'False':False,
            'j':1j
        }

        if s in special_cases:
            return special_cases[s]

        if s[0].isalpha():  # if s begins with a character in the alphabet return s as variable throws an error if incorrect format
            for l in s:
                try:
                    assert l.isalpha() or l.isdigit() or l=='_'
                except AssertionError:
                    raise Exception (l+': is not a valid variable')
            return s
        
        # Now assumes string can be transformed to a some constant
        iscomplex = s[-1]=='j'  # whether the number is complex or not
        s = s[:-1] if iscomplex else s

        if s.count('.')==1: # float and complex float cases
            return float(s)*1j if iscomplex else float(s)
        
        if s.isdigit():     # int and complex int cases
            return int(s)*1j if iscomplex else int(s)
        
        s = s+'j' if iscomplex else s   # If it has not been returned as a value or str throw an error
        raise Exception(f'{s}: is not recognized as a valid value or function')

    def list2tree(self,op_list:list):
        # Converts a tokenized expression to a tree
        # list2tree(['a','+','b'])=> node(val='+', left=node('a'), right=node('b'))
        # Handles parhentesis

        single_arg_operators = [
            'sin',
            'cos',
            'tan',
            'csc',
            'sec',
            'cot',
            'asin',
            'acos',
            'atan',
            '!',
            'exp',
            'ln']
        
        # Compress parhentesis protected expressions into sublists(sub expressions) within the full expression
        while '(' in op_list:
            op_list = self.compress_parhentesis(op_list)
        
        # expressions of length 1 are either variables, vals or a list element with a sub expression
        if len(op_list)==1:
            # variable or vals
            if type(op_list[0])in [str,bool, int, float, complex]:
                return node(op_list[0])
            # sub expressions
            elif type(op_list[0])==list:
                return self.list2tree(op_list[0])
        
        # expressions of length 2 are either a recognized single arg function or arbitrary function
        if len(op_list)==2:
            # negative operator is a special case that may occur
            if op_list[0]=='-':
                return node('*',node(-1),self.list2tree(op_list[1]))

            elif op_list[0] in single_arg_operators: # Recognized single argument expressions
                # returns op_list[0] operates on the expression or val of op_list[1]
                return node(op_list[0],right=self.list2tree(op_list[1]))
            
            # arbitrary function case
            elif isinstance(op_list[0],str) and isinstance(op_list[1],list):
                temp = ''.join(map(str,op_list[1][0]))      # recombines the list into a string
                temp = temp.split(',')                      # splits the string arguments seperated by commas
                return node(op_list[0],right = node(temp))  # returns a node of arbitrary operator operating on a node with val= to a list of arguments
        
        # special case of multiple operations led by a '-'
        # replaces -a... with -1*a...
        if op_list[0]=='-':
            op_list[0:2] = [[-1,'*',op_list[1]]] 

        # Determines the next operator which will operate left and right components
        n_op = next_operator(op_list)

        val = op_list[n_op]
        left = self.list2tree(op_list[:n_op])      # Splicing and recursion to assign left and right pointers
        right = self.list2tree(op_list[n_op+1:])   

        return node(val,left,right)

    
    def compress_parhentesis(self,exp_list):
        # algorithm that searches for the closing parhentesis of the first '('
        p1 = exp_list.index('(')   # left outermost parhentesis
        depth = 1
        for i,c in enumerate(exp_list[p1+1:]):
            if c=='(':
                depth+=1
            elif c==')':
                depth-=1
            
            if depth==0:
                break
        
        p2 = p1+i+1

        # returns a list with the first and outermost parhentesis surpressed expression enclosed by a list
        exp_list[p1:p2+1] = [[exp_list[p1+1:p2]]]   
        return exp_list

    def evaluate(self,root=None,val_dict:dict={}):
        # Evaluates an expression tree
        # If all the end nodes are operatable expressions returns value
        # Else returns None
        # Assumes the expression to be valid ie. For expressions with logical statements or equivalencies
        # it is assumed the expressions are valid

        '''
        >>> a = expr('a+b')
        >>> a.evaluate(val_dict={'a':1,'b':2})
        3
        '''

        if root==None:
            root =self.root
        
        if root!=None and isinstance(root.val,list): #arbitrary function case
            return None

        # For when the root val type is in a value set return the raw value
        if type(root.val) in [bool,int,float,float,complex]:
            return root.val
        
        elif isinstance(root.val,str) and root.right==None:# Identified variable type
            val_dict['pi'] = np.pi
            val_dict['e'] = np.e
            val_dict['i'] = 1j
            if root.val in val_dict:
                return val_dict[root.val]
            return None

        operator = {    # Operators with 2 inputs
            '+':lambda a,b: a+b,
            '-':lambda a,b: a-b,
            '/':lambda a,b: a/b if b!= 0 else None,
            '*':lambda a,b: a*b,
            '^':lambda a,b: a**b if (a and b)!=0 else None,
            '&':lambda a,b: a&b,
            '|':lambda a,b: a|b,
            '%':lambda a,b: a%b,
            '>':lambda a,b:a>b,
            '<':lambda a,b:a<b,
            '>=':lambda a,b:a>=b,   # need to recognize these operators within text
            '<=':lambda a,b:a<=b,
            '==': lambda a,b: a==b,
            '=': 'Easter Egg'
        }

        single_operators={          # operators with single inputs
            '!':lambda a:not a,
            'cos':lambda a: np.cos(a),
            'sin':lambda a: np.sin(a),
            'tan':lambda a: np.tan(a),
            'sec':lambda a: np.sec(a),
            'csc': lambda a: np.csc(a),
            'asin': lambda a: np.arcsin(a),
            'acos': lambda a: np.arccos(a),
            'atan':lambda a : np.arctan(a),
            'ln': lambda a: np.log(a),
            'exp': lambda a:np.exp(a)
        }

        # Given that val is an operation, node.right != None


        right = self.evaluate(root.right,val_dict=val_dict)
        
        # Special case of evaluating '='

        if root.val=='=': # '=' operator requires a little more complication
            left = self.evaluate(root.left, val_dict=val_dict)  # '=' has left node that can be evaluated

            # If both sides can be evaluated check equivalence
            if (left and right) in [bool,int,float,float,complex] and left!=right:
                raise Exception('Invalid expression')
            
            # Return left or right if they are valued datatypes
            elif type(left) in [bool,int,float,float,complex]:
                return left
            elif type(right) in [bool,int,float,float,complex]:
                return right
            
            return None

        if right==None: # Right must be real valued for an expression to be evaluated
            return None

        if root.val in single_operators:    # returns the respective value of the operation(value) for a single arg function
            return single_operators[root.val](right)

        left = self.evaluate(root.left, val_dict=val_dict)  # Evaluate left sub expression to evaluate 2 arg expression ie '+'

        if root.val in operator:
            if left ==None:
                return None
            return operator[root.val](left,right)   # maps root.val to the lambda operation in operator dict

    def display(self,root=None):
        # Purely for debugging purposes
        if root==None:
            root = self.root
        lines, *_ = self._display_aux(root)
        for line in lines:
            print(line)

    def _display_aux(self,base=None):
        # For debugging purposes
        
        if base.right== None and base.left==None:
            line = '%s' % base.val
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if base.right == None:
            lines, n, p, x = self._display_aux(base.left)
            s = '%s' % base.val
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if base.left == None:
            lines, n, p, x = self._display_aux(base.right)
            s = '%s' % base.val
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self._display_aux(base.left)
        right, m, q, y = self._display_aux(base.right)
        s = '%s' % base.val
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def estimate(self,var:str,precision:int=3):
        if precision>18:
            raise Exception('To many units of percision')
        r = np.random.rand()
        z = _copy(self.root)
        if z.val=='=':
            z = node('-',z.left,z.right)
        z = expr(root=z)
        dir = expr(root=z.root).pD(var)
        temp = expr(root = node('/',z.root,dir.root))
        for i in range(1000):
            r-=temp.evaluate(val_dict={var:r})

        d = dir.evaluate(val_dict={var:r})

        upper = r if d>0 else r-2*self.evaluate(val_dict={var:r})/d
        lower = r if d<0 else r-2*self.evaluate(val_dict={var:r})/d
        a = np.abs(d)*10
        a = 1/a

        while f'{upper:.{precision}e}'!= f'{lower:.{precision}e}':
            upper-=a*z.evaluate(val_dict={var:upper})*dir.evaluate(val_dict={var:upper})
            lower-=a*z.evaluate(val_dict={var:lower})*dir.evaluate(val_dict={var:lower})

        return float(f'{upper:.{precision-1}e}')
    
    def num_int(self,a,b,var,precision:int=3):
        if self.evaluate(val_dict={var:np.random.rand()})==None:
            raise Exception('Cannot perform numerical Integration')
        
        n = 200000
        l = self.left_Rsum(n,a,b,var)
        r = self.right_Rsum(n,a,b,var)
        counter = 0
        while f'{l:.{precision}e}'!=f'{r:.{precision}e}':
            n+=200000
            l = self.left_Rsum(n,a,b,var)
            r = self.right_Rsum(n,a,b,var)
        
        return float(f'{l:.{precision-1}e}')

    def right_Rsum(self,n,a,b,var):
        w = (b-a)/n
        s = 0
        x = a+w
        for i in range(n-1):
            x+=w
            s+= self.evaluate(val_dict={var:x})*w
        return s

    def left_Rsum(self,n,a,b,var):
        w = (b-a)/n
        s = 0
        x = a
        for i in range(n-1):
            x+=w
            s+= self.evaluate(val_dict={var:x})*w
        return s

    def map(self,base = None,path = []):
        # Sets the index of the variable to the value index
        # Returns a dict of 'variable' and [path] pairs
        # path is an ordered list of '1/0'
        # Going left and right for 1,0 respectfully will lead to 'variable'

        if base == None:
            base = self.root

        if base.right==None and isinstance(base.val,str):    # Case in which base is a variable save the current path to self.dir
            if base.val not in self.dir:
                self.dir[base.val] = []                      # Instantiate list for var
            
            if path not in self.dir[base.val]:      # only unique paths are added to self.dir for a given var
                self.dir[base.val].append(path)     # Append current path
            
        
        # Include arbitrary functions in self.dir
        arbitrary_function = base.right!=None and isinstance(base.right.val,list)

        if not arbitrary_function:
            if base.left is not None:
                self.map(base.left,path+[1])

            if base.right is not None:
                self.map(base.right,path+[0])
        
        else: # aritrary function
            if base.val not in self.dir:
                self.dir[base.val] = []        # Instantiate list for var
            self.dir[base.val].append(path)    # Append current path

    def __str__(self):
        # Returns a string expression that is an equivalent expression to the graph
        # str and __init__ should be rough inverses of each other
        # 'a$b'== str(exp('a$b'))
        """
        >>> print(expr('a+b'))
        a+b
        >>> print(expr('(a+b)*c'))
        (a+b)*c
        """
        # initial call to _str_aux
        return _str_aux(self.root)

    def invert_branch(self,var:str,include_var:bool=False):
        # Used for generating symbolic algebraic solutions to equations
        # inverts a particular path of the tree
        # generally used for inverting the path to a particular var
        # 'include_var'=True if the returned tree to be expressed as 'var = inv_tree'
        left_inv_dict = {
            '=':lambda inv_tree,right:node(
                '=',right,inv_tree
            ),
            '+':lambda inv_tree,right:node(
                '-',inv_tree,right
            ),
            '-':lambda inv_tree,right:node(
                '+',inv_tree,right
            ),
            '*':lambda inv_tree,right:node(
                '/',inv_tree,right
            ),
            '/':lambda inv_tree,right:node(
                '*',inv_tree,right
            ),
            '^':lambda inv_tree,right:node(
                '^',
                inv_tree,
                node(
                    '/',
                    node(1),
                    right
                )
            )
        }
        right_inv_dict={
            '=':lambda inv_tree,left:node(
                '=',left,inv_tree
            ),
            '+':lambda inv_tree,left:node(
                '-',inv_tree,left
            ),
            '-':lambda inv_tree,left:node(
                '-',left,inv_tree
            ),
            '*':lambda inv_tree,left:node(
                '/',inv_tree,left
            ),
            '/':lambda inv_tree,left:node(
                '/',left,inv_tree
            ),
            '^':lambda inv_tree,left:node(
                '/',
                node(
                    'ln',
                    right=inv_tree
                ),
                node(
                    'ln',
                    right=left
                )
            ),
            '!':lambda inv_tree,left:node(
                '!',
                right=inv_tree
            ),
            'exp':lambda inv_tree,left:node(
                'ln',
                right=inv_tree
            ),
            'ln':lambda inv_tree,left:node(
                'exp',
                right=inv_tree
            ),
            'sin':lambda inv_tree,left:node(
                'asin',
                right=inv_tree
            ),
            'cos':lambda inv_tree,left:node(
                'acos',
                right=inv_tree
            ),
            'tan':lambda inv_tree,left:node(
                'atan',
                right=inv_tree
            ),
            'csc':lambda inv_tree,left:node(
                'asin',
                right=node(
                    '/',
                    node(1),
                    inv_tree
                )
            ),
            'sec':lambda inv_tree,left:node(
                'acos',
                right=node(
                    '/',
                    node(1),
                    inv_tree
                )
            ),
            'cot':lambda inv_tree,left:node(
                'atan',
                right=node(
                    '/',
                    node(1),
                    inv_tree
                )
            ),
            'asin':lambda inv_tree,left:node(
                'sin',
                right=inv_tree
            ),
            'acos':lambda inv_tree,left:node(
                'cos',
                right=inv_tree
            ),
            'atan':lambda inv_tree,left:node(
                'tan',
                right=inv_tree
            )
        }
        
        root=self.root


        if var not in self.dir:
            raise Exception(f'\'{var}\' not found in Expression')

        path = self.dir[var][0] # if path is not specified takes the first path in dir
        C = 0 if self.evaluate()==None else self.evaluate() # seed to begin the inversion is either the current evaluation or 0
        
        # Case where the initial node is an equivalence operation
        if root.val =='=':
            inv_tree = root.right if path[0] else root.left
            root = root.left if path[0] else root.right
            path = path[1:]

        else:   # start the inversion with the seed C which is either 0 or self.evaluate()
            inv_tree = node(C)

        for d in path:
            if root.val not in right_inv_dict:
                print(f'Operator {root.val} not found in list of invertable expressions')
                return None

            if d:
                inv_tree = left_inv_dict[root.val](inv_tree,root.right)
                root=root.left
            else:
                inv_tree = right_inv_dict[root.val](inv_tree,root.left)
                root=root.right

        if include_var:
            inv_tree = node('=',node(var),inv_tree)
        
        inv_tree = reduce(inv_tree)
        return expr(root = inv_tree)

    def pD(self,var):
        # This process evaluates the derivative using dC/dx = 0, d(x)/dx = 1 and every mapping
        # of a d(f)/dx where f is a function of x
        root = self._partial_D_aux(self.root,var)
        root = reduce(root)
        root = common_form(root)        # reduce and common_form make the expression tree readable by trimming "extra" information
        root = reduce(root)
        return expr(root=root)
        
        
    def _partial_D_aux(self,root,var=''):
        if var=='':
            raise Exception(f'df/d"x" not defined in partial derivative')
        # a is the root of an expr with left and right components
        # below is a map of derivative rules written as da/dvar where left and right
        # nodes of a are assummed to be functions of var
        
        d_map = {                   # df/dx where f(x) #TODO convert map functions to discrete def's
            '+':lambda a,var: node( # f+g => f'+g'
                '+',self._partial_D_aux(a.left,var),self._partial_D_aux(a.right,var)),
            '-':lambda a,var: node( # f-g => f'-g'
                '-',self._partial_D_aux(a.left,var),self._partial_D_aux(a.right,var)),
            '*':lambda a,var: node( # f*g => f'g+fg'
                '+',
                left = node('*',self._partial_D_aux(a.left,var),a.right),
                right = node('*',a.left,self._partial_D_aux(a.right,var))),
            '/':lambda a,var:node(  # f/g => (f'*g-f*g')/(g^2)
                '/',
                left=node(
                  '-',
                  left = node('*',a.right,self._partial_D_aux(a.left,var)),
                  right = node('*',a.left,self._partial_D_aux(a.right,var))
                ),
                right = node('^',left = a.right,right=node(2))
            ),

            '^':lambda a,var:self._power_D(a,var), ## general formula for f^g was too hairy for lambda function

            'sin' :lambda a,var:node( # sin(f)=> f'*cos(f)
                '*',
                left = self._partial_D_aux(a.right,var),
                right = node('cos',right=a.right)
            ),
            'cos' :lambda a,var:node( # cos(f)=> -f'*sin(f)
                '*',
                left = node('*',node(-1),self._partial_D_aux(a.right,var)),
                right = node('sin',node(a.right))
            ),
            'tan' :lambda a,var:node( # tan(f)=> f'*sec(f)^2
                '*',
                left = self._partial_D_aux(a.right,var),
                right=node(
                    '^',
                    left = node('sec',right=a.right),
                    right = node(2))
            ),
            'csc' :lambda a,var:node( # csc(f)=> -f'*csc(f)*cot(f)
                '*',
                node(-1),
                node(
                    '*',
                    node(self._partial_D_aux(a.right,var)),
                    node(
                        '*',
                        node('csc',right=a),
                        node('cot',right=a)
                    )
                )
            ),
            'cot' :lambda a,var:node( # cot(f) => -f'*csc(f)^2
                '*',
                node(-1),
                node(
                    '*',
                    self._partial_D_aux(a.right,var),
                    node(
                        '^',
                        node('csc',right=a),
                        node(2)
                    )
                )
            ),
            'asin':lambda a,var:node( # asin(f) => f'/sqrt(1-f^2)
                node(
                    '/',
                    self._partial_D_aux(a.right,var),
                    node(
                        '^',
                        node(
                            '-',
                            node(1),
                            node('^',a,node(2))
                        ),
                        node(1/2)
                    )
                )
            ),
            'acos':lambda a,var:node( # acos(f) => -1*f'/sqrt(1-f^2)
                node(
                    '/',
                    node(
                        '*',
                        node(-1),
                        self._partial_D_aux(a.right,var)
                    ),
                    node(
                        '^',
                        node(
                            '-',
                            node(1),
                            node('^',a,node(2))
                        ),
                        node(1/2)
                    )
                )
            ),
            'atan':lambda a,var:node( # atan(f) => f'/(1+f^2)
                '/',
                self._partial_D_aux(a.right,var),
                node(
                    '+',
                    node(1),
                    node(
                        '^',
                        a.right,
                        node(2)
                    )
                )
            ),
            'exp' :lambda a,var:node( # exp(f) => f'*exp(f)
                '*',left=self._partial_D_aux(a.right,var),right=a),
            '='   :lambda a,var:node( # f=g => f'=g'
                '=',
                self._partial_D_aux(a.left,var),
                self._partial_D_aux(a.right,var)
            ),
            'ln':lambda a,var:node(
                '/',
                self._partial_D_aux(a.right,var),
                a.right
            )
        }

        if root.val==var:
            return node(1)
        
        elif root.right == None and root.val not in d_map or type(root) in [bool,int,float,complex]:
            return node(0)
        
        return d_map[root.val](root,var) 


    def _power_D(self,root,var):# general formula for df/dx(f^g) where f and g are functions of x
        f = root.left
        g = root.right
        
        # s1 = (g*f')/f
        s1 = node(
            '/',
            left = node(
                '*',
                g,
                self._partial_D_aux(f,var)
            ),
            right = f
        )
        # s2 = g'*ln(f)
        s2 = node(
            '*',
            left = self._partial_D_aux(g,var),
            right= node('ln',right=f)
        )

        s = node('+',s1,s2)

        return node('*',left = s,right=root)

    def common_form(self):
        return expr(root=common_form(self.root))

    def replace(self,var:str, sub):
        # sub can be a root, val or string describing an expression
        # nodes of var are replaced with the result of sub
        if type(sub) in [bool,int,float,complex]:
            sub = node(sub)
        elif isinstance(sub,str):
            sub = expr(sub).root
        
        self.dir = {}   # reinstantiates dir to have most current map
        self.map()
        if var in self.dir:
            for path in self.dir[var]:
                temp = self.root
                for left in path[:-1]:
                    temp = temp.left if left else temp.right

                if path[-1]:
                    temp.left =sub
                else:
                    temp.right = sub
        
    
    def simplify(self):
        pass

    def integrate(self,root,var):
        # TODO Not sure how to proceed with task
        # Integrating a constant
        # Demo of a process to perform a symbolic intgration

        special_cases = {
            1:node(var)
        }

        if root.var in special_cases:
            return special_cases[root.var]

        root = self.root
        
        if root.right==None:    # .right is only None in the case of a constant var
            
            pass
    
    def _integrate_expand(self, root,var):
        # expands an expression so that it can directly be mapped to an integral

        if type(root.val) in [int,float,complex]:
            return node(
                '+',
                node(
                    '*',
                    node(1),
                    node(var)
                ),
                node(
                    '*',
                    node(var),
                    node(0)
                )
            )

    def taylor_series(self,var,a,depth):
        # D is the expression that represents the current derivative
        # Taylor series is defined as Sum(D^n(f(a))/n!*(x-a)^n) :Where D is a derivative operator
        # Assumes xo is a leaf type
        if depth<1:
            raise Exception('Depth of taylor series needs to b greator than 1')
        
        temp = expr(root=_copy(self.root))          # 0th derivative of self
        next_temp = temp.pD(var)                    # 1st derivaive of self
        temp.replace(var,a)                         # In the copy of self replace var with a
        root = node(
            '+',
            temp.root,
            self._taylor_aux(next_temp,var,a,depth-1,1)
        )

        return expr(root = root)

    def _taylor_aux(self,f_prime,var,a,depth:int,n:int):
        
        temp = expr(root = _copy(f_prime.root))
        next_temp = temp.pD(var)
        temp.replace(var,a)
        # Constructs current term in the summation
        polynomial = node(
            '*',
            node('/',f_prime.root,node(_factorial(n))),
            node(
                '^',
                node('-',node(var),node(a)),
                node(n)
            )
        )
        if depth==0:
            return polynomial
        
        # Recursively calls the next term of the series
        return node(
            '+',
            polynomial,
            self._taylor_aux(next_temp,var,a,depth-1,n+1))

def _factorial(n):
    # factorial is built on 0!=1 and n! = n*(n-1)!
    if n==0:
        return 1
    return n*_factorial(n-1)

def _tokenize(input_str:str)->list:
    # Tokenize a string into a list of the macro elements of the exp
    # For each reserved command replace it with itself padded. 
    """
    >>> _tokenize('a+b')
    ['a', '+', 'b']
    >>> _tokenize('a==b')
    ['a', '==', 'b']
    >>> _tokenize('a==b=c')
    ['a', '==', 'b', '=', 'c']
    
    """
    double_operators = [
        '==',
        '<=',
        '>='
    ]
    exp_list = [
        '=',
        '(',
        ')',
        '+',
        '-',
        '*',
        '/',
        '^',
        '|',
        '&',
        '>',
        '<',
        '!',
        'ln',
        ','
    ]
    # trig functions and single argument expresssions
    # are implicitly tokenized if used correctly in an expression
    initial_tokens = []
    for e in double_operators:
        input_str = input_str.replace(e,' '+e+' ')
    initial_tokens = input_str.split(' ')
    tokenize_str = []
    for token in initial_tokens:
        if token in double_operators:
            tokenize_str.append(token)
            continue
        for e in exp_list:
            token = token.replace(e,' '+e+' ')
        tokenize_str+= token.split(' ')
    
    tokenize_str = [val for val in tokenize_str if val!='']

    return tokenize_str

def simplify(self,root):
    pass

def common_form(root):
    # Returns the root of a tree whose form follows
    # a+b+c+...
    # where a,b,c,... are of the form
    # d*e*f*...
    # where d,e,f,... are of the form
    # d^(g)
    # Where g is of common form
    root = _remove_minus_divide(root) # replace '-' with -1* and '/' with '^-1' to reduce sorting space
    root = distribute(root)
    roots2sum = _summed_terms(root)   # splices expression to a list of summands

    C = filter(lambda a:a!=None,[expr(root=term).evaluate() for term in roots2sum]) # Filter of summable terms
    C = sum(C)                                                                      # constant values are grouped to C
    roots2sum = [term for term in roots2sum if expr(root=term).evaluate()==None]    # Filter out terms that cannot be combined to C
    final_roots2sum = []
    if roots2sum==[]:       # returns C if it is the only term left
        return node(C)

    head_root = node('+')   # Instantiates the summation of terms
    flyer = head_root       # flyer 'flies' to the right creating a '+' series

    for summand in roots2sum:
        products = _product_terms(summand)
        
        var_products = []
        coefficient = 1     # Coefficient of summand term

        for c in products:  # Filters out terms that can be combined into the coefficient
            term = expr(root=c).evaluate()
            if term!=None:
                coefficient*=term
            else:
                var_products.append(c)
        
        var_products = [node('^',node('e'),n.right) if n.val=='exp' else n for n in var_products]   # replaces exp
        var_products = [p if p.val=='^' else node('^',p,node(1)) for p in var_products]

        bases = [p.left for p in var_products]
        powers = [p.right for p in var_products]
        
        unique_base_indexes = {}    # unique bases indexes and a [] of their powers

        for i,base in enumerate(bases):
            found = False
            for ui in unique_base_indexes:
                if equals(bases[ui],base):
                    found = True
                    unique_base_indexes[ui]+=[powers[i]]
                    break
            if not found:
                unique_base_indexes[i]=[powers[i]]
        
        # node_list = [node(coefficient)]
        node_list = []
        for ui in unique_base_indexes:
            power = _summation(unique_base_indexes[ui])
            power = common_form(power)
            node_list+=[node('^',bases[ui],power)]
        
        # indirect sort of node_list
        str_node_list = np.array([_str_aux(r) for r in node_list])
        i_sort = np.argsort(str_node_list)
        node_list = [node_list[i] for i in i_sort]  # sorts node list
        product = _product(node_list)

        combined = False
        for p in final_roots2sum:
            if equals(p.right,product):
                p.left.val+=coefficient
                combined = True
                break
        
        if combined:
            continue

        final_roots2sum+=[node('*',node(coefficient),product)]

    i_sort = np.argsort([_str_aux(n) for n in final_roots2sum])
    final_roots2sum = [final_roots2sum[i] for i in i_sort]
    temp = node('+',_summation(final_roots2sum),node(C))
    return reduce(temp)

def _str_aux(base,last_operator = None):
    # Auxillary equation of __str__
    # Referenced locally so technical parameters are hidden that are used for recursive calls

    op_order = {
        '=':1,
        '|':2,
        '&':3,
        '+':4,
        '-':4,
        '/':6,
        '*':5,
        '^':8
        }

    single_op = [
        '!',
        'sin',
        'cos',
        'tan',
        'csc',
        'sec',
        'cot',
        'asin',
        'acos',
        'atan',
        'ln',
        'exp'
    ]
    
    # Formats the tree to a string adding parhentesis to protect sub expressions as needed

    if base.val in single_op:
        return base.val+'('+_str_aux(base.right)+')'
    elif isinstance(base.val,str) and base.left==None and base.right!=None: # Arbitrary functions
        return base.val+'('+','.join(base.right.val)+')'

    if base.val not in op_order:
        return str(base.val)
    
    if last_operator and op_order[base.val]<op_order[last_operator]:
        return '('+ _str_aux(base.left,base.val)+base.val+_str_aux(base.right,base.val)+')'

    return _str_aux(base.left,base.val) + base.val + _str_aux(base.right,base.val)

def _summation(node_list):
    # special cases of empty and len==1 lists
    if node_list==[]:
        return node(0)
    elif len(node_list)==1:
        return node_list[0]
    
    head = node('+')
    head.left = node_list[0]
    flyer = head
    for n in node_list[1:-1]:
        flyer.right = node('+')
        flyer = flyer.right
        flyer.left = n

    flyer.right = node_list[-1]
    
    return head



def _product(node_list):
    if node_list==[]:
        return node(1)
    elif len(node_list)==1:
        return node_list[0]

    head = node('*')
    flyer = head
    for n in node_list:
        flyer.left = n
        flyer.right = node('*')
        flyer = flyer.right
    flyer.val = 1
    return head

def _summed_terms(root):
    if root==None:
        return []
    if root.val!='+':
        return [root]
    return _summed_terms(root.left)+_summed_terms(root.right)
def _product_terms(root):
    if root.val!='*':
        return [root]
    return _product_terms(root.left)+_product_terms(root.right)

def _copy(root):
    if root==None:
        return None
    return node(root.val,root.left,root.right)

def _remove_minus_divide(root):
    # Changes -(f)=> +(-1*f)
    # Changes /f => *f^(-1)
    if root==None:
        return None

    if root.val=='-':
        root.val='+'
        root.right = node(
            '*',
            node(-1),
            root.right
        )
    if root.val=='/':
        root.val='*'
        root.right = node(
            '^',
            root.right,
            node(-1)
        )
    
    return node(root.val,_remove_minus_divide(root.left),_remove_minus_divide(root.right))
    
def distribute(root):
    # After a remove_minus_divide
    # expressions of the form a*(b+c)=> a*b+a*c
    if root==None:
        return None
    
    if root.val=='*':
        if root.right.val=='+':
            root = node(
                '+',
                node('*',root.left,root.right.left),
                node('*',root.left,root.right.right)
            )
        elif root.left.val=='+':
            root= node(
                '+',
                node('*',root.right,root.left.left),
                node('*',root.right,root.left.right)
            )
    return node(root.val,distribute(root.left),distribute(root.right))

def equals(root1,root2):
    if root1==None and root2==None:
        return True
    elif root1.val!=root2.val:
        return False
    
    if equals(root1.left,root2.left) and equals(root1.right,root2.right):
        return True
    
    return False
        
def reduce(root):
    if root.right==None and isinstance(root.val,str): #instances of variables 
        return root
    elif type(root.val) in [int,bool,complex,float]:
        return root

    right = reduce(root.right)
    left = None
    if root.left:
        left = reduce(root.left)

    if root.val=='+':
        if left.val ==0:
            return right
        elif right.val == 0:
            return left
    elif root.val=='-':
        if right.val==0:
            return left
        elif left.val==0:
            return node('*',node(-1),right)
    
    elif root.val=='*':
        e = str(expr(root=root))

        if right.val==1:
            return left
        elif left.val==1:
            return right
        elif left.val==0 or right.val == 0:
            return node(0)

    elif root.val == '/':
        if right.val==1:
            return left
        if left.val==0:
            return left
        pass
    
    elif root.val=='exp':
        if right.val==0:
            return node(1)
        if right.val=='ln':
            return right.right
    
    elif root.val=='ln':
        if right.val=='e':
            return node(1)
        elif right.val=='^' and right.left.val=='e':
            return right.right
        elif right.val=='exp':
            return right.right            

    elif root.val=='^':
        if right.val==0 and left.val!=0:
            return node(0)
        elif left.val== (0) and right.val==(0):
            raise Exception('Invalid expression 0^0')
        elif left.val==1:
            return node(1)
        elif right.val==1:
            return left
        elif right.val==-1:
            return node('/',node(1),left)
        elif left.val=='e':
            return node('exp',right=right)
    
    elif root.val=='sin':
        if right==0:
            return right
    elif root.val=='cos':
        if right==0:
            return node(1)
    
    return node(root.val,left,right)

def next_operator(exp_list):
    # Returns the index of the macro operation
    # 'macro operation': if left and right components of the operation are grouped by parhentesis 
    # it wouldn't change the expression. 'a $ b' = '(a) $ (b)', where a and b are 
    # expressions and '$' is an operator
    # ex. 
    '''
    >>> from axioms_2 import *
    >>> next_operator('a+b*c')
    1
    '''
    operator = [
        '=',
        '|',
        '&',
        '!',
        '+',
        '-',
        '%',
        '*',
        '/',
        '^',
        '==',
        '<',
        '<=',
        '>',
        '>=',
        'cos',
        'sin',
        'tan',
        'sec',
        'csc',
        'cot',
        'asin',
        'acos',
        'atan',
        'ln']

    for op in operator:
        if op in exp_list:
            if op=='^':
                return exp_list.index('^')
            return len(exp_list)-exp_list[::-1].index(op)-1


def integrate(root,var):
    # If a form is recognized that can directly be integrated return the integrated form of the expression
    if root.right==None and root.val!=var:  #root is a real_value or not var
        # value_not_var => real value * var
        return node('*',node(root.val),node(var))

    if root.val==var:
        # var=> 1/2*(var)^2
        return node(
            '*',
            node(1/2),
            node(
                '^',
                node(var),
                node(2)
            )
        )
    
    if root.val=='+':
        return node(
            '+',
            integrate(root.left),
            integrate(root.right)
        )

    if root.val=='*':
        # need to create an algorithm that can check if function is of form f'*g'(f) ie chain rule
        pass


class node:
    # Units of expression objects
    # Describes the structure of mathematical expressions in with 
    # operations and left and right components

    def __init__(self,val,left:any= None,right:any = None):
        self.val = val
        self.right = right
        self.left = left


if __name__=="__main__":
    import doctest
    doctest.testmod()
    print(doctest.testfile('axioms_test.txt',report=True))