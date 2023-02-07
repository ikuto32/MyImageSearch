


/**
 * 
 * @param {(any, any) => number} f1 
 * @param {(any, any) => number} f2
 * @return {(any, any) => number}
 */
export function cancatComparator(f1, f2) {

    return (v1, v2) => {

        let out = f1(v1, v2)
        if(out != 0)
        {
            return out
        }

        return f2(v1, v2)
    }
}