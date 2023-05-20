import java.io.*;
import java.util.*;
import java.lang.*;
public class grazing
{
   static int[][] a = new int[5][5];
   static TreeSet<String> set = new TreeSet<>();
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("grazing.in"));
      for(int i = 0; i < 5; i++)
         for(int j = 0; j < 5; j++)
            a[i][j] = in.nextInt(); 
      in.close();
      
      for(int i = 0; i < 5; i++)
         for(int j = 0; j < 5; j++)
            solve(i, j, a[i][j] + "");
      
      System.out.println(set.size());
   }
   
   public static void solve(int i, int j, String s)
   {
      if(s.length() == 6)
      {
         set.add(s);
         return;
      }
      
      int[] di = {1, -1, 0, 0};
      int[] dj = {0, 0, -1, 1};
      
      for(int k = 0; k < 4; k++)
         if(i + di[k] < 5 && i + di[k] >= 0 && j + dj[k] < 5 && j + dj[k] >= 0)
            solve(i + di[k], j + dj[k], s + a[i + di[k]][j + dj[k]] + "");
   }
}