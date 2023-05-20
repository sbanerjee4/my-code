import java.util.*;
import java.io.*;
import java.lang.*;
public class farm
{
   static ArrayList[] cows;
   // static ArrayList[] regions;
   static int[] regions;
   static boolean[] marked;
   static int[] color;
   static int n, m;
   static String s;
   static int count = 0;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("farm.in"));
      n = in.nextInt();
      m = in.nextInt();
      s = in.next();
      cows = new ArrayList[n];
      // regions = new ArrayList[n];
      regions = new int[n];
      color = new int[n];
      marked = new boolean[n];
      
      for(int i = 0; i < n; i++)
         cows[i] = new ArrayList<Integer>();
      // for(int i = 0; i < n; i++)
         // regions[i] = new ArrayList<Integer>();
         
      for(int i = 0; i < n - 1; i++)
      {
         int a = in.nextInt() - 1, b = in.nextInt() - 1;
         cows[a].add(b);
         cows[b].add(a);
      }
      
      ff(0, false, 0);
      
      for(int i = 0; i < m; i++)
      {
         int a = in.nextInt() - 1, b = in.nextInt() - 1;
         int c = regions[a], d = regions[b];
         char ch = in.next().charAt(0);
        
         if(c == d)
            if(color[c] == (int)ch) System.out.print(1);
            else System.out.print(0);
         else System.out.print(1);
      }
      
      System.out.println();
   }
   
   static void ff(int pos, boolean newreg, int temp)
   {
      if(marked[pos]) 
         return;
      marked[pos] = true;
        
      if(newreg)
      {
         count++;
         temp = count;
      }
      color[temp] = (int)s.charAt(pos);
      regions[pos] = temp;
      for(int i = 0; i < cows[pos].size(); i++)
      {
         if(s.charAt((int)(cows[pos].get(i))) == s.charAt(pos))
            ff((int)(cows[pos].get(i)), false, temp);
         else
         {
            ff((int)(cows[pos].get(i)), true, temp);
         }
      }
   }
}