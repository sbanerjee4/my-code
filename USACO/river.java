import java.util.*;
import java.io.*;
import java.lang.*;
public class river
{
   static int[][] a = new int[52][52];
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("river.in"));
      int n = in.nextInt();
      for(int i = 0; i < n; i++)
      {
         int c1 = in.next().charAt(0);
         int c2 = in.next().charAt(0);
         if(Character.isUpperCase(c1)) c1 -= 'A';
         else c1 = c1 - 'a' + 26;
         
         if(Character.isUpperCase(c2)) c2 -= 'A';
         else c2 = c2 - 'a' + 26;
         
         int flow = in.nextInt();
         a[c1][c2] = flow;
      }
      in.close();
      
      for(int i = 0; i < 52; i++) reduce(i);
      for(int i = 0; i < 52; i++) cut(i);
      System.out.println(a[0][25]);
   }
   
   public static void cut(int pos)
   {
      if(pos == 0 || pos == 25) 
         return;
      for(int i = 0; i < 52; i++)
         if(a[i][pos] > 0) 
            return;
      for(int i = 0; i < 52; i++)
         if(a[pos][i] > 0)
         {
            a[pos][i] = 0;
            cut(i);
         }
   }
   
   public static void reduce(int pos)
   {
      if(pos == 0 || pos == 25) 
         return;
      int numParents = 0, numChildren = 0;
      for(int i = 0; i < 52; i++)
         if(a[i][pos] > 0) numParents++;
      for(int i = 0; i < 52; i++)
         if(a[pos][i] > 0) numChildren++;
      if(numParents != 1 || numChildren != 1) return;
      
      int child = 0, parent = 0;
      for(int i = 0; i < 52; i++)
        if(a[i][pos] > 0) parent = i;
      for(int i = 0; i < 52; i++)
        if(a[pos][i] > 0) child = i;
      
      a[parent][child] += Math.min(a[parent][pos], a[pos][child]);
      
      a[parent][pos] = 0;
      a[pos][child] = 0;
      reduce(parent);
      reduce(child);
   }
}