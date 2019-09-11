x = [1,2,3,4,5,6]
y = x
b = polyline(x,x, 'label')
bbb = [b,b]
x1 = chop(bbb, 1)
disp(x1(1))
def chop(obj, j, repeat=None):
    
    for k in range(len(obj)):
        if j > len(obj[k]):
            j = len(obj[k])
        result_1 = polyline(obj[k][0][:j], obj[k][1][:j])
        result_2 = []
        
        if repeat != None:
            if repeat:
                result_2 = polyline(obj[k][0][j:], obj[k][1][j:])
            else:
                result_2 = polyline(obj[k][0][j+1:], obj[k][0][j+1:])
                
    return result_1, result_2
                    
    
    